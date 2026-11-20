"""Verify BEVFusion geometry, fusion provenance and coverage without ground truth."""
import json
import pickle
from collections import Counter

import numpy as np

from fuse_three_models import ROOT, SOURCES, MAPPING3
from fuse_two_models_with_singletons import fingerprint
from two_model_detection import write_json


def covered(boxes, labels, target, iou_fn):
    result = np.zeros(len(boxes), dtype=bool)
    for cid in [1, 2, 3]:
        indices = np.flatnonzero(labels == cid)
        candidates = target[target[:, 7] == cid]
        if len(indices) and len(candidates):
            result[indices] = iou_fn(boxes[indices, :7].astype(np.float32), candidates[:, :7].astype(np.float32)).max(axis=1) >= .5
    return result


def main():
    from PIL import Image
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    output = ROOT / 'runs/three_model_detection'
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['frame_count'] == 204 and summary['two_model_baseline_exactly_reproduced']
    for name, digest in summary['source_sha256'].items():
        assert fingerprint(ROOT / name) == digest, name
    sources = []
    for path in [ROOT / f'runs/two_model_detection/{name}/result.pkl' for name in SOURCES[:2]]+[output / 'bevfusion/result.pkl']:
        with path.open('rb') as stream:
            sources.append(pickle.load(stream))
    with (output / 'fusion.pkl').open('rb') as stream:
        fused = pickle.load(stream)
    with (ROOT / 'runs/two_model_detection_singletons/fusion.pkl').open('rb') as stream:
        baseline = pickle.load(stream)
    with (output / 'bevfusion/native_geometry.pkl').open('rb') as stream:
        native = pickle.load(stream)
    metadata = json.loads((output / 'bevfusion/metadata.json').read_text())
    assert metadata['weights_loaded'] == metadata['weights_expected'] == 306
    assert metadata['frame_count'] == 204 and metadata['library_box_geometry_test'] == 'PASS'
    assert fingerprint(ROOT / metadata['checkpoint']) == metadata['checkpoint_sha256']
    transform = np.array(metadata['calibration'])
    ids = [row['frame_id'] for row in sources[0]]
    assert ids == list(fused) == list(baseline) == [r['frame_id'] for r in native]
    for source in sources:
        assert [r['frame_id'] for r in source] == ids
    config = summary['config']
    counts, provenance, aligned = np.zeros(3, dtype=int), Counter(), 0
    coverage = {name: Counter() for name in SOURCES}
    max_corner_error, max_enclosure_excess = 0.0, 0.0
    for index, frame_id in enumerate(ids):
        raw, standardized = native[index], sources[2][index]
        transformed = raw['corners_lidar'] @ transform[:3, :3].T+transform[:3, 3]
        max_corner_error = max(max_corner_error, float(np.max(abs(transformed-raw['corners_vehicle']))))
        expected_center = raw['gravity_center'] @ transform[:3, :3].T+transform[:3, 3]
        np.testing.assert_allclose(raw['converted_boxes'][:, :3], expected_center, atol=3e-5, rtol=1e-5)
        for points, box in zip(transformed, raw['converted_boxes']):
            yaw = box[6]
            basis = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            excess = abs((points-box[:3]) @ basis)-box[3:6]/2
            max_enclosure_excess = max(max_enclosure_excess, float(excess.max()))
            assert excess.max() < 3e-5
        selection = standardized['native_indices']
        np.testing.assert_array_equal(standardized['boxes_lidar'], raw['converted_boxes'][selection])
        np.testing.assert_array_equal(standardized['name'], raw['name'][selection])
        assert not np.isin(standardized['name'], ['barrier', 'traffic_cone']).any()
        record = fused[frame_id]
        boxes, refs, tags = record['gt_boxes'], record['source_indices'], record['provenance']
        assert boxes.shape == (len(tags), 9) and refs.shape == (len(tags), 3)
        assert np.isfinite(boxes).all() and (boxes[:, 3:6] > 0).all()
        for column in [0, 1, 2]:
            used = refs[:, column][refs[:, column] >= 0]
            assert len(set(used)) == len(used)
        for box, ref, tag in zip(boxes, refs, tags):
            parents, parent_sources = [], []
            for k, j in enumerate(ref):
                if j < 0:
                    continue
                row = sources[k][index]
                parent = np.r_[row['boxes_lidar'][j, :7], MAPPING3[row['name'][j]], row['score'][j]]
                assert int(parent[7]) == int(box[7]) and parent[8] >= config['admission_score']
                parents.append(parent)
                parent_sources.append(k)
            assert tag == '+'.join(SOURCES[k] for k in parent_sources)
            assert len(parents) >= 1
            for i, a in enumerate(parents):
                for b in parents[i+1:]:
                    cid = int(box[7])
                    assert np.linalg.norm(a[:2]-b[:2]) <= config['match_center_xy_m'][cid-1]+1e-6
                    assert abs(a[2]-b[2]) <= config['match_center_z_m']+1e-6
                    iou = iou3d_nms_utils.boxes_bev_iou_cpu(a[None, :7].astype(np.float32), b[None, :7].astype(np.float32))[0, 0]
                    assert iou >= config['match_bev_iou'][cid-1]-1e-6
            values = np.array(parents)
            strongest = values[np.argmax(values[:, 8])]
            expected = strongest.copy()
            if len(values) == 1:
                assert box[8] >= config['singleton_score']-1e-6
            else:
                dims = values[:, 3:6].copy()
                if 2 in parent_sources:
                    for i, parent in enumerate(values):
                        delta = parent[6]-strongest[6]
                        if abs(np.sin(delta)) > abs(np.cos(delta)):
                            dims[i, :2] = dims[i, :2][::-1]
                            aligned += 1
                expected[3:6] = (dims*values[:, 8, None]).sum(axis=0)/values[:, 8].sum()
                expected[8] = (values[:, 8]**2).sum()/values[:, 8].sum()
            assert box[8] >= config['output_score']-1e-6
            np.testing.assert_allclose(box, expected, atol=1e-5, rtol=1e-5)
        counts += [(boxes[:, 7] == cid).sum() for cid in [1, 2, 3]]
        provenance.update(tags)
        with Image.open(output / 'comparison' / f'{frame_id}.png') as image:
            image.load()
            assert image.size == (1980, 1320)
        for name, rows in zip(SOURCES, sources):
            row = rows[index]
            selected = row['score'] >= .7
            high = row['boxes_lidar'][selected]
            labels = np.array([MAPPING3[n] for n in row['name'][selected]])
            before = covered(high, labels, baseline[frame_id]['gt_boxes'], iou3d_nms_utils.boxes_bev_iou_cpu)
            after = covered(high, labels, boxes, iou3d_nms_utils.boxes_bev_iou_cpu)
            coverage[name].update(high_score_inputs=len(high), covered_two=int(before.sum()), covered_three=int(after.sum()),
                                  newly_covered=int((~before & after).sum()), no_longer_covered=int((before & ~after).sum()))
    assert counts.tolist() == summary['totals']['three_model_fusion']
    assert dict(provenance) == summary['provenance']
    report = dict(status='PASS', frame_count=204, independent_counts=counts.tolist(), provenance=dict(provenance),
                  source_sha256_unchanged=True, baseline_two_models_exactly_reproduced=True,
                  max_transformed_corner_error_m=max_corner_error,
                  max_corner_outside_upright_enclosure_numerical_error_m=max_enclosure_excess,
                  members_requiring_axis_alignment_in_retained_groups=aligned,
                  high_score_prediction_coverage={k: dict(v) for k, v in coverage.items()},
                  coverage_definition='Original predictions score>=0.7, same supercategory BEV IoU>=0.5, many-to-one. Not ground truth.',
                  checks=['306/306 BEVFusion checkpoint keys', 'Actual legacy bottom/yaw corner conversion',
                          'Upright enclosure contains all transformed corners', 'Frame order/coverage',
                          'Source box used at most once', 'All member pairs satisfy matching gates',
                          'Independent axis-aligned fusion geometry/score', 'Singleton confidence',
                          '204 decoded PNGs', 'Output validity and independent counts', 'Original SHA256 preserved'])
    write_json(output / 'verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
