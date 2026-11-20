"""Independent BCV source/fusion checks and full final pseudo-label verification."""
import argparse
import json
from collections import Counter

import numpy as np

import run_ms3d_from_three_model_fusion as bridge
from run_bcv_ms3d import ROOT, FUSION, REFINED, SOURCE_ORDER, SOURCE_PATHS, CP
from fuse_three_models import MAPPING3
from fuse_two_models_with_singletons import fingerprint
from two_model_detection import write_json


def verify_fusion():
    from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu
    summary = json.loads((FUSION / 'summary.json').read_text())
    manifest = json.loads(bridge.MANIFEST.read_text())
    ids = [f['frame_id'] for f in manifest['frames']]
    data, fusion = [bridge.load(p) for p in SOURCE_PATHS], bridge.load(FUSION / 'fusion.pkl')
    assert list(fusion) == ids
    for rows in data:
        assert [r['frame_id'] for r in rows] == ids
    for path, digest in {**summary['source_sha256'], **summary['prepared_source_sha256']}.items():
        assert fingerprint(ROOT / path) == digest, path
    for path, digest in json.loads((FUSION / 'baseline_sha256.json').read_text()).items():
        assert fingerprint(ROOT / path) == digest, path
    config = summary['config']
    counts, provenance = np.zeros(3, dtype=int), Counter()
    aligned, cp_checked = 0, 0
    for index, fid in enumerate(ids):
        original_cp = json.loads((CP / 'predictions' / f'{fid}.json').read_text())
        names, original = np.array(original_cp['names']), np.array(original_cp['boxes'])
        keep = np.flatnonzero(np.isin(names, list(MAPPING3)))
        converted = data[1][index]
        np.testing.assert_array_equal(converted['original_indices'], keep)
        np.testing.assert_array_equal(converted['name'], names[keep])
        expected = original[keep][:, [0,1,2,3,4,5,8]].copy()
        expected[:, 2] += 1.8
        np.testing.assert_allclose(converted['boxes_lidar'], expected, atol=4e-6, rtol=1e-6)
        np.testing.assert_allclose(converted['score'], np.array(original_cp['scores'])[keep], atol=1e-8)
        cp_checked += len(keep)
        record = fusion[fid]
        boxes, refs, tags = record['gt_boxes'], record['source_indices'], record['provenance']
        assert boxes.shape == (len(tags),9) and refs.shape == (len(tags),3)
        assert np.isfinite(boxes).all() and (boxes[:,3:6] > 0).all()
        assert np.isin(boxes[:,7], [1,2,3]).all()
        for col in range(3):
            used = refs[:,col][refs[:,col] >= 0]
            assert len(used) == len(set(used))
        for box, ref, tag in zip(boxes, refs, tags):
            parents, sources = [], []
            for k,j in enumerate(ref):
                if j >= 0:
                    source = data[k][index]
                    assert j < len(source['score'])
                    parents.append(np.r_[source['boxes_lidar'][j,:7], MAPPING3[source['name'][j]], source['score'][j]])
                    sources.append(k)
            values = np.array(parents)
            assert len(values) and (values[:,7] == box[7]).all()
            assert tag == '+'.join(SOURCE_ORDER[k] for k in sources)
            assert (values[:,8] >= config['admission_score']).all()
            cid = int(box[7])-1
            for a in range(len(values)):
                for b in range(a+1,len(values)):
                    x,y = values[a], values[b]
                    assert np.linalg.norm(x[:2]-y[:2]) <= config['match_center_xy_m'][cid]+1e-5
                    assert abs(x[2]-y[2]) <= config['match_center_z_m']+1e-5
                    overlap = boxes_bev_iou_cpu(x[None,:7].astype(np.float32),y[None,:7].astype(np.float32))[0,0]
                    assert overlap >= config['match_bev_iou'][cid]-1e-5
            expected = values[np.argmax(values[:,8])].copy()
            if len(values) == 1:
                assert box[8] >= config['singleton_score']-1e-6
            else:
                dims = values[:,3:6].copy()
                delta = values[:,6]-expected[6]
                swap = abs(np.sin(delta)) > abs(np.cos(delta))
                aligned += int(swap.sum())
                dims[swap,:2] = dims[swap,:2][:,::-1]
                expected[3:6] = (dims*values[:,8,None]).sum(axis=0)/values[:,8].sum()
                expected[8] = (values[:,8]**2).sum()/values[:,8].sum()
            np.testing.assert_allclose(box,expected,atol=1e-5,rtol=1e-5)
            assert box[8] >= config['output_score']-1e-6
        for cid in [1,2,3]:
            selected = boxes[boxes[:,7] == cid]
            if len(selected)>1:
                matrix = boxes_bev_iou_cpu(selected[:,:7],selected[:,:7])
                np.fill_diagonal(matrix,0)
                assert matrix.max() <= config['final_nms_bev_iou'][cid-1]+3e-5
        counts += [(boxes[:,7] == c).sum() for c in [1,2,3]]
        provenance.update(tags)
    assert counts.tolist() == summary['totals']['fusion']
    assert dict(provenance) == summary['provenance']
    report = dict(status='PASS',frames=len(ids),counts=counts.tolist(),provenance=dict(provenance),
                  centerpoint_boxes_transform_checked=cp_checked,axis_alignments_in_retained_groups=aligned,
                  original_source_and_baseline_hashes_unchanged=True,
                  checks=['Source frame order','CenterPoint conversion and class exclusion independently checked',
                          'Unique source use','Every member pair passes gates','Independent center/yaw/size/score formula',
                          'Singleton thresholds','Final fusion NMS geometry','Original/derived source SHA256'])
    write_json(FUSION / 'verification.json',report)
    print('FUSION VERIFIED',json.dumps(report),flush=True)


def verify_final():
    import verify_ms3d_pseudo_labels as verifier
    verifier.OUT, verifier.SOURCE = REFINED, FUSION / 'fusion.pkl'
    verifier.main()
    initial = bridge.load(REFINED / 'initial_pseudo_labels.pkl')
    final = bridge.load(REFINED / 'final_ps_dict.pkl')
    native = bridge.load(REFINED / 'final_native_two_class.pkl')
    checked = 0
    for fid in final:
        mask = abs(final[fid]['gt_boxes'][:,7]) != 3
        for key in ['gt_boxes','num_pts','memory_counter']:
            np.testing.assert_array_equal(final[fid][key][mask], native[fid][key])
        # Fusion already applied the identical cyclist NMS0.3; compute support
        # independently to verify that eligible cyclists are neither dropped nor invented.
        candidates = initial[fid]['gt_boxes'][abs(initial[fid]['gt_boxes'][:,7]) == 3].copy()
        points = np.load(ROOT / 'runs/two_model_detection/prepared' / f'{fid}.npy')
        keep = []
        for box in candidates:
            d = points-box[:3]
            c,s = np.cos(box[6]),np.sin(box[6])
            support = (abs(d[:,0]*c+d[:,1]*s) <= box[3]/2+.1) & (abs(-d[:,0]*s+d[:,1]*c) <= box[4]/2+.1) & (abs(d[:,2]) <= box[5]/2)
            if support.sum()>1 and not(abs(box[0])<1 and abs(box[1])<1):
                box[7] = 3 if box[8]>=.5 else -3
                keep.append(box)
        actual = final[fid]['gt_boxes'][abs(final[fid]['gt_boxes'][:,7]) == 3]
        # NMS may reorder disjoint retained rows by score; compare independently sorted tuples.
        expected = np.array(keep).reshape(-1,9)
        a_order = np.lexsort(actual[:,:3].T) if len(actual) else []
        e_order = np.lexsort(expected[:,:3].T) if len(expected) else []
        np.testing.assert_allclose(actual[a_order], expected[e_order],atol=1e-7,rtol=0)
        checked += len(actual)
    report = dict(status='PASS',native_vehicle_pedestrian_unchanged=True,
                  cyclist_boxes_checked=checked,cyclist_temporal_refinement=False)
    write_json(REFINED / 'cyclist_verification.json',report)
    print('CYCLIST VERIFIED',json.dumps(report),flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--phase',choices=['all','fusion','final'],default='all')
    args = p.parse_args()
    if args.phase in ['all','fusion']:
        verify_fusion()
    if args.phase in ['all','final']:
        verify_final()


if __name__ == '__main__':
    main()
