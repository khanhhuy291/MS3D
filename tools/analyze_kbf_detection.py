"""Trace box removal and run fusion-only ablations on saved detector outputs."""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcdet.utils import box_fusion_utils as fusion
from pcdet.ops.iou3d_nms import iou3d_nms_utils

OUT = ROOT / 'runs/two_model_detection'
RADII = [1.5, .3, .2]
NMS = [.1, .3, .1]
MAPPING = {'car': 1, 'truck': 1, 'bus': 1, 'pedestrian': 2, 'motorcycle': 3, 'bicycle': 3}


def covered(boxes, candidates):
    if not len(boxes) or not len(candidates):
        return np.zeros(len(boxes), dtype=bool)
    return iou3d_nms_utils.boxes_bev_iou_cpu(
        boxes[:, :7].astype(np.float32), candidates[:, :7].astype(np.float32)).max(axis=1) >= .5


def fuse_before_nms(boxes, radius):
    groups = fusion.get_matching_boxes(boxes, discard=2, radius=radius)
    candidates, used, sizes = [], set(), Counter()
    for group in groups:
        matched = boxes[list(group)]
        if len(np.unique(matched[:, :3].round(4), axis=0)) < 2:
            continue
        used.update(group)
        sizes[len(group)] += 1
        result = fusion.kbf(matched[:, :9], matched[:, 8])
        if result is not None:
            candidates.append(result)
    return np.array(candidates, dtype=np.float32).reshape(-1, 9), used, sizes


def nms(boxes, threshold):
    if len(boxes) < 2:
        return boxes
    return boxes[fusion.nms(boxes[:, :7], boxes[:, 8], thresh=threshold)]


def main():
    sources = {}
    for name in ['lyft', 'nuscenes']:
        with (OUT / name / 'result.pkl').open('rb') as stream:
            sources[name] = pickle.load(stream)
    with (OUT / 'fusion/kbf.pkl').open('rb') as stream:
        saved = pickle.load(stream)
    stats = {name: Counter() for name in sources}
    stages, group_sizes = Counter(), Counter()
    variants = {name: np.zeros(3, dtype=int) for name in
                ['baseline', 'vehicle_nms_0.5', 'admission_0.1', 'admission_0.1_vehicle_nms_0.5']}
    examples = []
    for frame_index, rows in enumerate(zip(*sources.values())):
        if rows[0]['frame_id'] != rows[1]['frame_id']:
            raise ValueError('Frame mismatch')
        frame_id = rows[0]['frame_id']
        all_boxes, origin = [], []
        for source_id, row in enumerate(rows):
            class_ids = np.array([MAPPING[n] for n in row['name']])
            all_boxes.append(np.column_stack([row['boxes_lidar'], class_ids, row['score'], np.ones(len(class_ids))]))
            origin.extend([source_id]*len(class_ids))
        all_boxes = np.vstack(all_boxes)
        origin = np.array(origin)
        for class_id in [1, 2, 3]:
            class_mask = all_boxes[:, 7] == class_id
            raw = all_boxes[class_mask]
            raw_origin = origin[class_mask]
            keep = raw[:, 8] > .3
            boxes, source_ids = raw[keep], raw_origin[keep]
            candidates, used, sizes = fuse_before_nms(boxes, RADII[class_id-1])
            group_sizes.update(sizes)
            final = nms(candidates, NMS[class_id-1])
            expected = saved[frame_id]['gt_boxes']
            expected = expected[expected[:, 7] == class_id]
            # Compare unordered outputs to prove the traced baseline matches the real run.
            def sorted_rows(x):
                return x[np.lexsort((x[:, 8], x[:, 2], x[:, 1], x[:, 0]))]
            np.testing.assert_allclose(sorted_rows(final), sorted_rows(expected), rtol=1e-5, atol=1e-5)
            stages.update(input_boxes=len(boxes), input_boxes_in_groups=len(used),
                          input_boxes_without_group=len(boxes)-len(used),
                          candidate_groups=len(candidates), final_boxes=len(final),
                          candidates_removed_by_nms=len(candidates)-len(final))
            variants['baseline'][class_id-1] += len(final)
            relaxed = nms(candidates, .5 if class_id == 1 else NMS[class_id-1])
            variants['vehicle_nms_0.5'][class_id-1] += len(relaxed)
            for source_id, name in enumerate(sources):
                indices = np.flatnonzero(source_ids == source_id)
                selected = boxes[indices]
                pre_cover, final_cover = covered(selected, candidates), covered(selected, final)
                isolated = np.array([i not in used for i in indices], dtype=bool)
                stats[name].update(input_boxes=len(indices), no_group=int(isolated.sum()),
                                   no_group_score_ge_0_7=int((isolated & (selected[:, 8] >= .7)).sum()),
                                   no_group_score_ge_0_9=int((isolated & (selected[:, 8] >= .9)).sum()),
                                   covered_pre_nms_iou_0_5=int(pre_cover.sum()),
                                   covered_final_iou_0_5=int(final_cover.sum()),
                                   covered_pre_but_not_final=int((pre_cover & ~final_cover).sum()),
                                   in_group_but_not_covered_pre=int((~isolated & ~pre_cover).sum()))
                low_partner = raw[(raw_origin != source_id) & (raw[:, 8] <= .3)]
                strong_partner = boxes[source_ids != source_id]
                for local_i in np.flatnonzero(isolated):
                    box = selected[local_i]
                    has_low = bool(len(low_partner) and np.any(np.linalg.norm(low_partner[:, :3]-box[:3], axis=1) <= RADII[class_id-1]))
                    overlap_partner = bool(covered(box[None], strong_partner)[0])
                    stats[name]['isolated_with_low_score_partner'] += int(has_low)
                    stats[name]['isolated_but_other_model_bev_iou_ge_0_5'] += int(overlap_partner)
                    if frame_id == 'frame_00100' and box[8] >= .7 and class_id == 1:
                        examples.append(dict(frame_id=frame_id, model=name, center=box[:3].tolist(),
                                             score=float(box[8]), low_score_partner=has_low,
                                             overlapping_partner_outside_radius=overlap_partner))
            low = raw[raw[:, 8] > .1]
            low_candidates, _, _ = fuse_before_nms(low, RADII[class_id-1])
            for key, threshold in [('admission_0.1', NMS[class_id-1]),
                                   ('admission_0.1_vehicle_nms_0.5', .5 if class_id == 1 else NMS[class_id-1])]:
                result = nms(low_candidates, threshold)
                variants[key][class_id-1] += int((result[:, 8] >= .3).sum())
        if (frame_index+1) % 40 == 0:
            print(f'Traced {frame_index+1}/204 frames', flush=True)
    report = dict(frame_count=204, baseline_matches_saved=True, stages=dict(stages),
                  group_size_histogram=dict(group_sizes), per_model={k: dict(v) for k, v in stats.items()},
                  fusion_only_ablations={k: v.tolist() for k, v in variants.items()},
                  high_score_isolated_examples_frame_100=examples,
                  caveats=['Counts are box-frame, not unique objects or accuracy.',
                           'Coverage uses same-supercategory BEV IoU >= 0.5, many-to-one.',
                           'Candidate NMS removals include duplicates from overlapping groups.',
                           'More outputs in an ablation do not imply better precision or recall.'])
    dest = OUT / 'kbf_diagnosis'
    dest.mkdir(exist_ok=True)
    (dest / 'diagnosis.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
