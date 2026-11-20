"""Artifact-contract tests for the read-only local visualization server."""
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ms3d_local_server', ROOT / 'web/local_viewer/server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class ViewerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = server.ViewerStore(ROOT)

    def test_catalog_has_expected_frames_cases_and_coordinate_frame(self):
        catalog = self.store.catalog()
        self.assertEqual(len(catalog['frames']), 204)
        self.assertEqual(catalog['coordinate_frame'], 'vehicle')
        self.assertEqual(catalog['default_case'], 'fusion3')
        self.assertEqual({c['id'] for c in catalog['cases']},
                         {'lyft', 'nuscenes', 'bevfusion', 'kbf2', 'fusion2', 'fusion3', 'fusion2_relaxed', 'ms3d_final', 'centerpoint', 'fusion_bcv', 'ms3d_bcv'})

    def test_bcv_source_order_and_final_verified_class_counts(self):
        fusion = json.loads((ROOT / 'runs/fusion_bcv/verification.json').read_text())
        final = json.loads((ROOT / 'runs/ms3d_bcv_round1/verification.json').read_text())
        for key, expected in [('fusion_bcv',fusion['counts']),
                              ('ms3d_bcv',[final['positive_stats'][n] for n in ['vehicle','pedestrian','cyclist']])]:
            records = [d for frame in self.store.records[key] for d in frame]
            self.assertEqual([sum(d['class_id'] == c for d in records) for c in [1,2,3]],expected)
            if key == 'fusion_bcv':
                sources = set(s for d in records for s in d['sources'])
                self.assertEqual(sources,{'voxelrcnn','centerpoint','bevfusion'})
                self.assertTrue(all('lyft' not in d['sources'] and 'nuscenes' not in d['sources'] for d in records))
            else:
                self.assertTrue(all(not d['sources'] for d in records))
                self.assertTrue(all(d['score'] >= {1:.7,2:.6,3:.5}[d['class_id']] for d in records))

    def test_centerpoint_all_frames_transform_fields_classes_and_totals(self):
        from collections import Counter
        root = ROOT / 'runs/centerpoint_voxel1440_fm_singleframe'
        summary = json.loads((root / 'summary.json').read_text())
        verification = json.loads((root / 'verification.json').read_text())
        counts = Counter()
        total = 0
        for i, frame in enumerate(self.store.frames):
            source = json.loads((root / 'predictions' / f"{frame['frame_id']}.json").read_text())
            actual = self.store.records['centerpoint'][i]
            boxes = np.array([d['box'] for d in actual])
            raw = np.array(source['boxes'])
            # Independently verify the recorded frame relation: Z_vehicle=Z_model+1.8;
            # XY/dimensions/length-axis yaw already have the target convention.
            np.testing.assert_allclose(boxes[:, :2], raw[:, :2], atol=1e-12)
            np.testing.assert_allclose(boxes[:, 2], raw[:, 2]+1.8, atol=1e-12)
            np.testing.assert_array_equal(boxes[:, 3:6], raw[:, 3:6])
            np.testing.assert_array_equal(boxes[:, 6], raw[:, 8])
            self.assertEqual([d['score'] for d in actual], source['scores'])
            self.assertEqual([d['original_name'] for d in actual], source['names'])
            for d in actual:
                self.assertEqual(d['class_id'], server.LABELS[d['original_name']])
                self.assertEqual(d['source_indices'], {'centerpoint': d['index']})
                if d['score'] >= .3:
                    counts[d['original_name']] += 1
            total += len(actual)
        self.assertEqual(total, verification['total_boxes_score_gt_01'])
        self.assertEqual(sum(counts.values()), verification['total_boxes_score_ge_03'])
        self.assertEqual(dict(counts), summary['displayed_class_counts'])

    def test_final_pseudo_labels_match_verified_positive_export(self):
        expected = json.loads((ROOT / 'runs/ms3d_three_model_round1/verification.json').read_text())['positive_stats']
        detections = [d for frame in self.store.records['ms3d_final'] for d in frame]
        self.assertEqual(len(detections), expected['box_count'])
        for class_id, name, threshold in [(1, 'vehicle', .7), (2, 'pedestrian', .6)]:
            selected = [d for d in detections if d['class_id'] == class_id]
            self.assertEqual(len(selected), expected[name])
            self.assertTrue(all(d['score'] >= threshold for d in selected))
        self.assertTrue(all(d['kind'] == 'unknown' and not d['sources'] for d in detections))
        self.assertFalse(next(c for c in self.store.cases if c['id'] == 'ms3d_final')['has_provenance'])

    def test_normalized_boxes_match_independently_verified_run_totals(self):
        for key, path, count_key in [
            ('fusion3', 'runs/three_model_detection/summary.json', 'three_model_fusion'),
            ('fusion2', 'runs/two_model_detection_singletons/summary.json', 'pair_singletons')]:
            summary = json.loads((ROOT / path).read_text())
            expected = summary['totals'][count_key]
            actual = [sum(d['class_id'] == c for frame in self.store.records[key] for d in frame) for c in [1, 2, 3]]
            self.assertEqual(actual, expected)

    def test_binary_points_remain_in_prepared_vehicle_frame(self):
        data, count, original_count = self.store.points(100, 1000)
        array = np.frombuffer(data, dtype='<f4').reshape(-1, 3)
        expected = np.load(ROOT / 'runs/two_model_detection/prepared/frame_00100.npy')
        indices = np.linspace(0, len(expected)-1, 1000).astype(np.int64)
        np.testing.assert_array_equal(array, expected[indices])
        self.assertEqual(count, 1000)
        self.assertEqual(original_count, len(expected))

    def test_out_of_range_frame_and_point_budget_are_rejected(self):
        for index in [-1, 204]:
            with self.assertRaises(ValueError):
                self.store.frame(index)
        with self.assertRaises(ValueError):
            self.store.points(0, 999999)

    def test_provenance_distinguishes_single_pair_triple_and_unknown(self):
        kinds = set()
        for detections in self.store.records['fusion3']:
            for detection in detections:
                kinds.add(detection['kind'])
                self.assertEqual(len(detection['sources']), {'single':1, 'pair':2, 'triple':3}[detection['kind']])
                self.assertEqual(set(detection['sources']), set(detection['source_indices']))
        self.assertEqual(kinds, {'single', 'pair', 'triple'})
        self.assertTrue(all(d['kind'] == 'unknown' for d in self.store.records['kbf2'][0]))


if __name__ == '__main__':
    unittest.main()
