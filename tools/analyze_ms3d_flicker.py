"""Read-only continuity audit of saved MS3D proposals versus final labels.

Coverage is geometric (same-class BEV IoU), not final track lineage or GT.
Overlapping branches may count the same physical object more than once.
"""
import json
from collections import Counter, defaultdict

import numpy as np

from run_ms3d_from_three_model_fusion import OUT, MANIFEST, load
from two_model_detection import write_json


def main():
    from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu
    from pcdet.utils.transform_utils import world_to_ego, ego_to_world
    from scipy.optimize import linear_sum_assignment

    manifest = json.loads(MANIFEST.read_text())['frames']
    poses = {f['frame_id']: np.array(f['pose']) for f in manifest}
    initial = load(OUT / 'initial_pseudo_labels.pkl')
    final = load(OUT / 'final_ps_dict.pkl')
    audits = defaultdict(list)
    for row in json.loads((OUT / 'point_support_audit.json').read_text()):
        audits[row['frame_id']].append(row)

    def best(box, boxes):
        if not len(boxes):
            return 0., -1
        values = boxes_bev_iou_cpu(np.array(box[:7], dtype=np.float32)[None],
                                  np.array(boxes, dtype=np.float32)[:, :7])[0]
        index = int(values.argmax())
        return float(values[index]), index

    result = dict(method='Same-class best BEV IoU >= threshold; diagnostic geometric coverage, not GT or one-to-one lineage',
                  caveat='Counts are track-frame slots; overlapping branches are not unique objects.', thresholds={})
    for threshold in [.3, .5]:
        groups = {}
        for name, path, key, cid in [
            ('veh_all', 'veh_all_after_merge.pkl', 'frameid_to_box', 1),
            ('veh_static', 'veh_static_after_propagation.pkl', 'frameid_to_propboxes', 1),
            ('ped', 'ped_after_refinement.pkl', None, 2),
        ]:
            tracks = load(OUT / 'stages' / path)
            statuses, holes, rows = Counter(), [], []
            skipped_dynamic = 0
            for tid, track in tracks.items():
                if name == 'veh_static' and track['motion_state'] != 0:
                    skipped_dynamic += 1
                    continue
                mapping = track[key] if key else dict(zip(track['frame_id'], track['boxes']))
                states = []
                for fid, world_box in sorted(mapping.items()):
                    _, boxes = world_to_ego(poses[fid], boxes=np.array(world_box)[None])
                    box = boxes[0]
                    target = final[fid]['gt_boxes']
                    positive = target[target[:, 7] == cid]
                    ignored = target[target[:, 7] == -cid]
                    coverage, index = best(box, positive)
                    ignore_coverage, _ = best(box, ignored)
                    candidates = [r for r in audits[fid] if abs(r['box'][7]) == cid]
                    point_coverage, pi = best(box, [r['box'] for r in candidates])
                    ped_guard = cid == 2 and not (abs(initial[fid]['gt_boxes'][:, 7]) == 2).any()
                    if coverage >= threshold:
                        status = 'positive_present'
                    elif ignore_coverage >= threshold:
                        status = 'ignore_only'
                    elif ped_guard:
                        status = 'ped_initial_empty_guard'
                    elif point_coverage >= threshold and candidates[pi]['num_points'] <= 1:
                        status = 'point_support_rejection'
                    else:
                        status = 'no_positive_geometric_match_after_nms'
                    row = dict(frame_id=fid, track_id=int(tid), status=status,
                               best_positive_bev_iou=coverage, best_ignore_bev_iou=ignore_coverage)
                    if index >= 0:
                        selected = positive[index]
                        row.update(final_score=float(selected[8]),
                                   center_difference_m=float(np.linalg.norm(box[:3]-selected[:3])))
                    states.append(row)
                    statuses[status] += 1
                present = [i for i, r in enumerate(states) if r['status'] == 'positive_present']
                if len(present) > 1:
                    for a, b in zip(present, present[1:]):
                        if b > a+1:
                            missing = states[a+1:b]
                            holes.append(dict(track_id=int(tid), before=states[a]['frame_id'],
                                              after=states[b]['frame_id'], missing_count=len(missing),
                                              reasons=dict(Counter(r['status'] for r in missing)),
                                              missing_frames=[r['frame_id'] for r in missing]))
                rows.extend(states)
            holes.sort(key=lambda h: (-h['missing_count'], h['before']))
            groups[name] = dict(track_count=len(tracks)-skipped_dynamic, statuses=dict(statuses),
                                internal_missing_runs=len(holes), internal_missing_slots=sum(h['missing_count'] for h in holes),
                                internal_reason_counts=dict(sum((Counter(h['reasons']) for h in holes), Counter())),
                                internal_holes=holes)
            if threshold == .3:
                write_json(OUT / f'flicker_{name}_slots.json', rows)
        result['thresholds'][str(threshold)] = groups
    # Inspect final positives outside the retained track support, and short
    # appearances using independent neighboring-frame geometric association.
    world_positive, supported, positive_rows = {}, {}, {}
    for fid in final:
        boxes = final[fid]['gt_boxes']
        boxes = boxes[boxes[:, 7] > 0]
        positive_rows[fid] = boxes
        _, world_positive[fid] = ego_to_world(poses[fid], boxes=boxes)
        supported[fid] = np.zeros(len(boxes), dtype=bool)
    for group, cid in [('veh_all', 1), ('veh_static', 1), ('ped', 2)]:
        path, key = {
            'veh_all': ('veh_all_after_merge.pkl', 'frameid_to_box'),
            'veh_static': ('veh_static_after_propagation.pkl', 'frameid_to_propboxes'),
            'ped': ('ped_after_refinement.pkl', None),
        }[group]
        tracks = load(OUT / 'stages' / path)
        for slot in json.loads((OUT / f'flicker_{group}_slots.json').read_text()):
            if slot['status'] != 'positive_present':
                continue
            fid, tid = slot['frame_id'], slot['track_id']
            # Slot support is recalculated against all targets, not only the best.
            track = tracks[tid]
            mapping = track[key] if key else dict(zip(track['frame_id'], track['boxes']))
            _, ego = world_to_ego(poses[fid], boxes=np.array(mapping[fid])[None])
            targets = positive_rows[fid]
            if len(targets):
                overlaps = boxes_bev_iou_cpu(ego[:, :7].astype(np.float32), targets[:, :7].astype(np.float32))[0]
                supported[fid] |= (targets[:, 7] == cid) & (overlaps >= .5)
    previous = {fid: np.zeros(len(b), dtype=bool) for fid,b in world_positive.items()}
    following = {fid: np.zeros(len(b), dtype=bool) for fid,b in world_positive.items()}
    fids = list(world_positive)
    for first, second in zip(fids, fids[1:]):
        for cid in [1,2,3]:
            a, b = world_positive[first], world_positive[second]
            ai, bi = np.flatnonzero(a[:,7] == cid), np.flatnonzero(b[:,7] == cid)
            if not len(ai) or not len(bi):
                continue
            matrix = boxes_bev_iou_cpu(a[ai,:7].astype(np.float32), b[bi,:7].astype(np.float32))
            r, c = linear_sum_assignment(np.where(matrix >= .3, -(1+min(matrix.shape)+matrix), 1e6))
            keep = matrix[r,c] >= .3
            following[first][ai[r[keep]]] = True
            previous[second][bi[c[keep]]] = True
    aggregate, examples = Counter(), []
    for fid in fids:
        boxes = positive_rows[fid]
        aggregate['positive_total'] += len(boxes)
        aggregate['positive_supported_by_kept_track_iou_0p5'] += int(supported[fid].sum())
        aggregate['positive_without_kept_track_support_iou_0p5'] += int((~supported[fid]).sum())
        if fid not in [fids[0], fids[-1]]:
            islands = ~previous[fid] & ~following[fid]
            aggregate['single_frame_geometric_islands_iou_0p3'] += int(islands.sum())
            aggregate['islands_without_kept_track_support_iou_0p5'] += int((islands & ~supported[fid]).sum())
            for j in np.flatnonzero(islands):
                examples.append(dict(frame_id=fid, final_index=int(j), box=boxes[j].tolist(),
                                     kept_track_support=bool(supported[fid][j])))
    result['positive_coverage'] = dict(aggregate)
    result['single_frame_examples'] = examples[:30]
    print('POSITIVE COVERAGE', json.dumps(result['positive_coverage']))
    write_json(OUT / 'flicker_diagnosis.json', result)
    for threshold, groups in result['thresholds'].items():
        print('IoU', threshold)
        for name, group in groups.items():
            print(name, json.dumps({k:v for k,v in group.items() if k!='internal_holes'}))
            print('Examples:', json.dumps(group['internal_holes'][:3]))


if __name__ == '__main__':
    main()
