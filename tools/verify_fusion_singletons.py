"""Independent artifact, provenance and source-coverage checks for singleton fusion."""
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from fuse_two_models_with_singletons import ROOT, fingerprint
from two_model_detection import MAPPING, write_json


def coverage(boxes, labels, targets, iou_fn):
    result = np.zeros(len(boxes), dtype=bool)
    for class_id in [1, 2, 3]:
        source_ids = np.flatnonzero(labels == class_id)
        candidates = targets[targets[:, 7] == class_id]
        if len(source_ids) and len(candidates):
            ious = iou_fn(boxes[source_ids, :7].astype(np.float32), candidates[:, :7].astype(np.float32))
            result[source_ids] = ious.max(axis=1) >= .5
    return result


def main():
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    from PIL import Image
    original, output = ROOT / 'runs/two_model_detection', ROOT / 'runs/two_model_detection_singletons'
    summary = json.loads((output / 'summary.json').read_text())
    for name, expected in summary['source_sha256'].items():
        if fingerprint(ROOT / name) != expected:
            raise ValueError(f'Original artifact changed: {name}')
    source = {}
    for name in ['lyft', 'nuscenes']:
        with (original / name / 'result.pkl').open('rb') as stream:
            source[name] = pickle.load(stream)
    with (original / 'fusion/kbf.pkl').open('rb') as stream:
        old = pickle.load(stream)
    with (output / 'fusion.pkl').open('rb') as stream:
        fused = pickle.load(stream)
    expected_ids = [row['frame_id'] for row in source['lyft']]
    assert expected_ids == [row['frame_id'] for row in source['nuscenes']] == list(fused) == list(old)
    assert len(fused) == summary['frame_count'] == 204
    totals, provenance, stats = np.zeros(3, dtype=int), Counter(), {k: Counter() for k in source}
    example = None
    for index, frame_id in enumerate(expected_ids):
        row = fused[frame_id]
        boxes, refs, origin = row['gt_boxes'], row['source_indices'], row['provenance']
        assert boxes.shape == (len(origin), 9) and refs.shape == (len(origin), 2)
        assert np.isfinite(boxes).all() and (boxes[:, 3:6] > 0).all()
        assert (boxes[:, 8] >= summary['config']['output_score']-1e-6).all()
        for column in [0, 1]:
            used = refs[:, column][refs[:, column] >= 0]
            assert len(set(used)) == len(used), 'A source box was used more than once'
        sources = [source[name][index] for name in ['lyft', 'nuscenes']]
        for box, ref, kind in zip(boxes, refs, origin):
            parents = []
            for column, parent_index in enumerate(ref):
                if parent_index < 0:
                    continue
                parent = sources[column]
                label = MAPPING[parent['name'][parent_index]]
                assert label == int(box[7])
                assert parent['score'][parent_index] >= summary['config']['admission_score']
                parents.append(np.r_[parent['boxes_lidar'][parent_index, :7], label, parent['score'][parent_index]])
            if kind == 'matched':
                assert len(parents) == 2
                a, b = parents
                class_id = int(box[7])
                assert np.linalg.norm(a[:2]-b[:2]) <= summary['config']['match_center_xy_m'][class_id-1]+1e-6
                assert abs(a[2]-b[2]) <= summary['config']['match_center_z_m']+1e-6
                pair_iou = iou3d_nms_utils.boxes_bev_iou_cpu(a[None, :7].astype(np.float32), b[None, :7].astype(np.float32))[0, 0]
                assert pair_iou >= summary['config']['match_bev_iou'][class_id-1]-1e-6
                strong = a if a[8] >= b[8] else b
                expected = strong.copy()
                expected[3:6] = (a[3:6]*a[8]+b[3:6]*b[8])/(a[8]+b[8])
                expected[8] = (a[8]**2+b[8]**2)/(a[8]+b[8])
                np.testing.assert_allclose(box, expected, atol=1e-5, rtol=1e-5)
            else:
                assert kind in ['lyft_singleton', 'nuscenes_singleton'] and len(parents) == 1
                assert (kind == 'lyft_singleton') == (ref[0] >= 0)
                assert parents[0][8] >= summary['config']['singleton_score']
                np.testing.assert_allclose(box, parents[0], atol=1e-5, rtol=1e-5)
        totals += [(boxes[:, 7] == c).sum() for c in [1, 2, 3]]
        provenance.update(origin)
        with Image.open(output / 'comparison' / f'{frame_id}.png') as image:
            image.load()
            assert image.size == (2640, 660) and image.format == 'PNG'
        for name, row_source in zip(['lyft', 'nuscenes'], sources):
            mask = row_source['score'] >= .7
            high_boxes = row_source['boxes_lidar'][mask]
            labels = np.array([MAPPING[n] for n in row_source['name'][mask]])
            old_cover = coverage(high_boxes, labels, old[frame_id]['gt_boxes'], iou3d_nms_utils.boxes_bev_iou_cpu)
            new_cover = coverage(high_boxes, labels, boxes, iou3d_nms_utils.boxes_bev_iou_cpu)
            stats[name].update(high_score_inputs=len(high_boxes), covered_original=int(old_cover.sum()),
                               covered_new=int(new_cover.sum()), newly_covered=int((~old_cover & new_cover).sum()),
                               no_longer_covered=int((old_cover & ~new_cover).sum()))
        if frame_id == 'frame_00100':
            target = np.array([3.5663283, 6.1370087, .7856268])
            near = np.flatnonzero(np.linalg.norm(boxes[:, :3]-target, axis=1) < .01)
            assert len(near) == 1, 'The previously lost score .96 example was not restored'
            k = int(near[0])
            example = dict(frame_id=frame_id, box=boxes[k].tolist(), provenance=origin[k])
    assert totals.tolist() == summary['totals']['pair_singletons']
    assert dict(provenance) == summary['provenance']
    report = dict(status='PASS', frame_count=len(fused), independent_totals=totals.tolist(),
                  provenance=dict(provenance), high_score_input_bev_coverage={k: dict(v) for k, v in stats.items()},
                  restored_example=example, original_sha256_unchanged=True,
                  coverage_definition='Original detector boxes score>=0.7; same-supercategory BEV IoU>=0.5, many-to-one. Not ground truth or recall.',
                  checks=['Frame coverage/order', 'Finite output and positive dimensions',
                          'Unique use of each source box', 'Cross-model pair provenance and all matching gates',
                          'Independent pair geometry/score calculation', 'Singleton identity and score threshold',
                          'Decode and verify all 204 PNGs', 'Totals and provenance recount',
                          'Original artifacts SHA256 unchanged', 'Known lost .96 example restored'])
    write_json(output / 'verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
