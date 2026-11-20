"""Verify the native MS3D++ output and measure changes against stored fusion."""
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from run_ms3d_from_three_model_fusion import ROOT, OUT, SOURCE, MANIFEST, load, label_stats
from fuse_two_models_with_singletons import fingerprint
from two_model_detection import write_json


def main():
    from scipy.optimize import linear_sum_assignment
    from pcdet.ops.iou3d_nms import iou3d_nms_utils as iou
    initial, final, positive = [load(OUT / name) for name in ['initial_pseudo_labels.pkl','final_ps_dict.pkl','final_positive_only.pkl']]
    original = load(SOURCE)
    metadata = json.loads((OUT / 'run_metadata.json').read_text())
    import yaml
    config = yaml.safe_load((OUT / 'ps_config.yaml').read_text())
    summary = json.loads((OUT / 'summary.json').read_text())
    manifest = json.loads(MANIFEST.read_text())
    expected_ids = [f['frame_id'] for f in manifest['frames']]
    assert list(original) == list(initial) == list(final) == list(positive) == expected_ids
    for path, digest in metadata['source_sha256'].items():
        assert fingerprint(ROOT / path) == digest, path
    canonical_min_iou, checked, swaps = 1.0, 0, 0
    numeric_point_differences, shifts, changes, per_frame = [], [], Counter(), []
    for index, frame_id in enumerate(expected_ids):
        a, b = original[frame_id]['gt_boxes'], initial[frame_id]['gt_boxes']
        assert a.shape == b.shape and np.isfinite(b).all()
        np.testing.assert_array_equal(a[:,[0,1,2,5,7,8]],b[:,[0,1,2,5,7,8]])
        if len(a):
            equivalent = np.diag(iou.boxes_bev_iou_cpu(a[:,:7].astype(np.float32),b[:,:7].astype(np.float32)))
            canonical_min_iou = min(canonical_min_iou,float(equivalent.min()))
            assert (equivalent >= .9999).all()
        swaps += int(((a[:,7] == 1)&(a[:,3] < a[:,4])).sum())
        frame = final[frame_id]
        boxes = frame['gt_boxes']
        assert boxes.ndim == 2 and boxes.shape[1] == 9 and np.isfinite(boxes).all()
        valid_classes = [-1,-2,-3,1,2,3] if metadata.get('preserve_cyclists', False) else [-1,-2,1,2]
        assert (boxes[:,3:6] > 0).all() and np.isin(boxes[:,7],valid_classes).all()
        assert len(frame['num_pts']) == len(frame['memory_counter']) == len(boxes)
        assert (frame['memory_counter'] == 0).all()
        assert not ((abs(boxes[:,0]) < 1)&(abs(boxes[:,1]) < 1)).any()
        thresholds = np.array(config['PS_SCORE_TH']['POS_TH'])[abs(boxes[:,7]).astype(int)-1]
        assert (boxes[boxes[:,7] > 0,8] >= thresholds[boxes[:,7] > 0]-1e-6).all()
        assert (boxes[boxes[:,7] < 0,8] < thresholds[boxes[:,7] < 0]+1e-6).all()
        mask = boxes[:,7] > 0
        for key in ['gt_boxes','num_pts','memory_counter']:
            np.testing.assert_array_equal(positive[frame_id][key],frame[key][mask])
        points = np.load(ROOT / 'runs/two_model_detection/prepared' / f'{frame_id}.npy')
        x,y,z = points.T
        for row, recorded in zip(boxes,frame['num_pts']):
            dx,dy,dz = x-row[0],y-row[1],z-row[2]
            c,s = float(np.cos(float(row[6]))),float(np.sin(float(row[6])))
            u,v = dx*c+dy*s,-dx*s+dy*c
            contained=(abs(u)<=row[3]/2+.1)&(abs(v)<=row[4]/2+.1)&(abs(dz)<=row[5]/2)
            count=int(contained.sum())
            if count != int(recorded):
                numeric_point_differences.append(dict(frame_id=frame_id,computed=count,recorded=int(recorded)))
            assert count > 1 and int(recorded) > 1
            checked += 1
        # One-to-one geometric correspondence to describe output changes, not GT matching.
        current = positive[frame_id]['gt_boxes']
        frame_changes = Counter()
        for cid in [1,2,3]:
            old = b[b[:,7] == cid]
            new = current[current[:,7] == cid]
            if len(old) and len(new):
                matrix=iou.boxes_bev_iou_cpu(new[:,:7].astype(np.float32),old[:,:7].astype(np.float32))
                rows,cols=linear_sum_assignment(1-matrix)
                accepted=matrix[rows,cols] >= .1
                rows,cols=rows[accepted],cols[accepted]
                for ni,oi in zip(rows,cols):
                    delta=new[ni,:3]-old[oi,:3]
                    shifts.append(dict(frame_id=frame_id,class_id=cid,center_shift_m=float(np.linalg.norm(delta)),
                                       center_shift_xy_m=float(np.linalg.norm(delta[:2])),center_shift_z_m=float(abs(delta[2])),
                                       input_box=old[oi].tolist(),final_box=new[ni].tolist()))
                frame_changes['positive_matched_to_initial']+=len(rows)
                frame_changes['positive_without_initial_iou_0p1']+=len(new)-len(rows)
            else:
                frame_changes['positive_without_initial_iou_0p1']+=len(new)
        changes.update(frame_changes)
        per_frame.append(dict(frame_id=frame_id,**frame_changes))
        if (index+1)%40==0:
            print(f'Verified {index+1}/204 frames',flush=True)
    assert swaps == metadata['vehicle_axis_swaps']
    assert label_stats(final) == summary['final'] and label_stats(positive) == summary['final_positive_only']
    # Point counts are expected to agree exactly for this recorded run.
    assert not numeric_point_differences, numeric_point_differences[:5]
    propagation=load(OUT / 'stages/veh_static_after_propagation.pkl')
    max_extension=0
    for track in propagation.values():
        base=track['frameid_to_box']
        first=min(expected_ids.index(fid) for fid in base)
        last=max(expected_ids.index(fid) for fid in base)
        for fid,box in track['frameid_to_propboxes'].items():
            assert fid in initial and np.isfinite(box).all()
            position=expected_ids.index(fid)
            extension=max(first-position,position-last,0)
            max_extension=max(max_extension,extension)
            assert extension<=10
    native_files=[ROOT / 'pcdet/utils'/name for name in ['ms3d_utils.py','tracker_utils.py','transform_utils.py','compatibility_utils.py','box_fusion_utils.py']]
    native_files+=list((ROOT / 'tracker/mot_3d').rglob('*.py'))
    native_files+=[ROOT / 'pcdet/datasets/augmentor/augmentor_utils.py']
    write_json(OUT / 'code_fingerprints.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in native_files})
    distance=np.array([row['center_shift_m'] for row in shifts])
    deltas=dict(matched_count=len(shifts),mean_m=float(distance.mean()) if len(distance) else 0,
                median_m=float(np.median(distance)) if len(distance) else 0,
                p95_m=float(np.percentile(distance,95)) if len(distance) else 0,max_m=float(distance.max()) if len(distance) else 0)
    write_json(OUT / 'output_changes.json',dict(correspondence='Same class, Hungarian BEV IoU >=0.1; relative to input predictions, not GT',
                                             counts=dict(changes),center_shift=deltas,
                                             largest_shifts=sorted(shifts,key=lambda x:-x['center_shift_m'])[:20],per_frame=per_frame))
    report=dict(status='PASS',frame_count=204,checked_final_boxes=checked,
                canonical_vehicle_swaps=swaps,canonical_footprint_min_bev_iou=canonical_min_iou,
                point_support_counts_exact=True,point_margin_xy_m=.1,max_external_propagation_frames=max_extension,
                source_sha256_unchanged=True,positive_stats=label_stats(positive),final_stats=label_stats(final),
                output_changes=dict(changes),center_shift_against_matched_initial=deltas,
                checks=['204 matching frame IDs/order','Input canonicalization preserves footprint/center/score',
                        'Positive dimensions and finite values','Independent point support counts on all final boxes',
                        'Class-ID sign matches POS_TH','Positive-only export exactly matches signed native output',
                        'No ego-center boxes','Propagation within10frames and sequence',
                        'Source fusion/manifest/config SHA256 preserved'])
    write_json(OUT / 'verification.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
