"""Add BEVFusion to the previous source-aware fusion, preserving a controlled baseline."""
import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from fuse_two_models_with_singletons import ROOT, DEFAULT_CONFIG, fingerprint, fuse_pair, select_pairs
from two_model_detection import CLASSES, MAPPING, write_json

SOURCES = ['lyft', 'nuscenes', 'bevfusion']
MAPPING3 = dict(MAPPING, construction_vehicle=1, trailer=1)


def class_rows(row, class_id, admission):
    labels = np.array([MAPPING3[n] for n in row['name']])
    ids = np.flatnonzero((labels == class_id) & (row['score'] >= admission))
    return np.column_stack([row['boxes_lidar'][ids, :7], labels[ids], row['score'][ids]]), ids


def gates(a, b, iou_fn):
    ious = iou_fn(a[:, :7].astype(np.float32), b[:, :7].astype(np.float32))
    xy = np.linalg.norm(a[:, None, :2]-b[None, :, :2], axis=2)
    z = abs(a[:, None, 2]-b[None, :, 2])
    return ious, xy, z


def fuse_members(members, align_all_pairs=False):
    boxes = np.array([m['box'] for m in members])
    if len(boxes) == 1:
        return boxes[0].astype(np.float32)
    if len(boxes) == 2 and not align_all_pairs and not any(m['source'] == 2 for m in members):
        return fuse_pair(boxes[0], boxes[1])
    weights = boxes[:, 8]
    result = boxes[np.argmax(weights)].copy()
    # A rectangle admits equivalent (dx,dy,yaw) and (dy,dx,yaw+pi/2)
    # representations. Align each member's axes to the strongest member before
    # averaging sizes; BEVFusion legacy boxes often use width as the first edge.
    aligned_dims = boxes[:, 3:6].copy()
    diff = boxes[:, 6]-result[6]
    swap = abs(np.sin(diff)) > abs(np.cos(diff))
    aligned_dims[swap, :2] = aligned_dims[swap, :2][:, ::-1]
    result[3:6] = np.average(aligned_dims, axis=0, weights=weights)
    result[8] = np.average(boxes[:, 8], weights=weights)
    return result.astype(np.float32)


def fuse_frame(rows, config, iou_fn, nms_fn):
    source_names = config.get('source_order', SOURCES)
    final, refs_out, provenance_out = [], [], []
    stats = Counter()
    for class_id in [1, 2, 3]:
        arrays, original_ids = zip(*(class_rows(row, class_id, config['admission_score']) for row in rows))
        a, b = arrays[:2]
        def member(source, index):
            return dict(source=source, index=int(original_ids[source][index]), box=arrays[source][index])
        if len(a) and len(b):
            pair = select_pairs(*gates(a, b, iou_fn), config['match_bev_iou'][class_id-1],
                                config['match_center_xy_m'][class_id-1], config['match_center_z_m'])
        else:
            pair = np.empty((0, 2), dtype=int)
        groups = [[member(0, i), member(1, j)] for i, j in pair]
        for source in [0, 1]:
            used = set(pair[:, source])
            groups.extend([[member(source, i)] for i in range(len(arrays[source])) if i not in used])
        stats[f'initial_{source_names[0]}_{source_names[1]}_pairs'] += len(pair)
        if len(rows) == 3:
            third = arrays[2]
            if groups and len(third):
                # Every member must agree with the third model, preventing bridging.
                all_members = np.array([m['box'] for group in groups for m in group])
                all_iou, all_xy, all_z = gates(all_members, third, iou_fn)
                offsets = np.cumsum([0]+[len(g) for g in groups])
                min_iou = np.stack([all_iou[start:end].min(axis=0) for start, end in zip(offsets[:-1], offsets[1:])])
                max_xy = np.stack([all_xy[start:end].max(axis=0) for start, end in zip(offsets[:-1], offsets[1:])])
                max_z = np.stack([all_z[start:end].max(axis=0) for start, end in zip(offsets[:-1], offsets[1:])])
                assignments = select_pairs(min_iou, max_xy, max_z, config['match_bev_iou'][class_id-1],
                                           config['match_center_xy_m'][class_id-1], config['match_center_z_m'])
            else:
                assignments = np.empty((0, 2), dtype=int)
            for group_index, third_index in assignments:
                stats[f'{source_names[2]}_joined_existing_pair' if len(groups[group_index]) == 2 else f'{source_names[2]}_joined_single_source'] += 1
                groups[group_index].append(member(2, third_index))
            used = set(assignments[:, 1])
            groups.extend([[member(2, i)] for i in range(len(third)) if i not in used])
        candidates, refs, tags = [], [], []
        for group in groups:
            box = fuse_members(group, align_all_pairs=config.get('align_all_pairs', False))
            threshold = config['singleton_score'] if len(group) == 1 else config['output_score']
            stats[f'groups_size_{len(group)}_before_score'] += 1
            if box[8] < max(threshold, config['output_score']):
                stats[f'groups_size_{len(group)}_rejected_score'] += 1
                continue
            candidates.append(box)
            ref = [-1]*len(rows)
            for m in group:
                ref[m['source']] = m['index']
            refs.append(ref)
            tags.append('+'.join(source_names[i] for i, x in enumerate(ref) if x >= 0))
        candidates = np.array(candidates, dtype=np.float32).reshape(-1, 9)
        keep = (np.flatnonzero(nms_fn(candidates[:, :7], candidates[:, 8], config['final_nms_bev_iou'][class_id-1]))
                if len(candidates) > 1 else np.arange(len(candidates)))
        stats['candidates_before_nms'] += len(candidates)
        stats['removed_by_nms'] += len(candidates)-len(keep)
        for i in keep:
            final.append(candidates[i])
            refs_out.append(refs[i])
            provenance_out.append(tags[i])
    return dict(gt_boxes=np.array(final, dtype=np.float32).reshape(-1, 9),
                source_indices=np.array(refs_out, dtype=int).reshape(-1, len(rows)),
                provenance=provenance_out), stats


def common_mask(boxes, extrinsic):
    xyz = boxes[:, :3]
    inverse = np.linalg.inv(extrinsic)
    lidar = xyz @ inverse[:3, :3].T + inverse[:3, 3]
    return (((lidar >= [-54, -54, -5]) & (lidar < [54, 54, 3])).all(axis=1)
            & ((xyz >= [-75.2, -75.2, -2]) & (xyz < [75.2, 75.2, 4])).all(axis=1))


def raw_boxes(row, threshold):
    mask = row['score'] >= threshold
    labels = np.array([MAPPING3[n] for n in row['name'][mask]])
    return np.column_stack([row['boxes_lidar'][mask, :7], labels, row['score'][mask]])


def render_frame(frame_id, panels, prepared, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    points = np.load(prepared / f'{frame_id}.npy')
    points = points[(abs(points[:, 0]) < 40) & (abs(points[:, 1]) < 40)][::2]
    colors = ['#00d2ff', '#ffcf33', '#ff66bb']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
    for ax, (name, boxes, tags) in zip(axes.flat, panels):
        ax.set_facecolor('#101822')
        ax.scatter(points[:, 0], points[:, 1], s=.2, c='#8998a8', alpha=.6, rasterized=True)
        segments, line_colors, styles = [], [], []
        for box, tag in zip(boxes, tags):
            x, y, z, dx, dy, dz, yaw, cid, score = box
            corners = np.array([[dx/2, dy/2], [dx/2, -dy/2], [-dx/2, -dy/2], [-dx/2, dy/2], [dx/2, dy/2]])
            rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            segments.extend([corners @ rot.T+[x, y], np.array([[x, y], [x+dx/2*np.cos(yaw), y+dx/2*np.sin(yaw)]])])
            line_colors.extend([colors[int(cid)-1]]*2)
            styles.extend(['dashed' if tag == 'singleton' else 'solid']*2)
        ax.add_collection(LineCollection(segments, colors=line_colors, linewidths=.8, linestyles=styles))
        ax.set(xlim=(-40, 40), ylim=(-40, 40), title=f'{name}: {len(boxes)} boxes',
               xlabel='vehicle X forward (m)', ylabel='vehicle Y left (m)')
        ax.set_aspect('equal')
    fig.suptitle(f'{frame_id} | output score >=0.3 | vehicle frame | fusion: dashed=single source, solid=2/3 sources')
    fig.tight_layout()
    fig.savefig(destination, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    from pcdet.utils.box_fusion_utils import nms
    output = ROOT / 'runs/three_model_detection'
    source_paths = [ROOT / 'runs/two_model_detection' / name / 'result.pkl' for name in SOURCES[:2]]
    source_paths += [output / 'bevfusion/result.pkl']
    baseline_path = ROOT / 'runs/two_model_detection_singletons/fusion.pkl'
    original_path = ROOT / 'runs/two_model_detection/fusion/kbf.pkl'
    input_paths = source_paths+[baseline_path, original_path, ROOT / 'runs/two_model_detection/manifest.json']
    hashes = {str(path.relative_to(ROOT)): fingerprint(path) for path in input_paths}
    data = []
    for path in source_paths:
        with path.open('rb') as stream:
            data.append(pickle.load(stream))
    with baseline_path.open('rb') as stream:
        baseline = pickle.load(stream)
    with original_path.open('rb') as stream:
        old_kbf = pickle.load(stream)
    manifest = json.loads(input_paths[-1].read_text())
    expected_ids = [f['frame_id'] for f in manifest['frames']]
    for rows in data:
        if [r['frame_id'] for r in rows] != expected_ids:
            raise ValueError('Three-model frame order/coverage mismatch')
    extrinsic = np.array(manifest['extrinsic'])
    config = dict(DEFAULT_CONFIG, source_order=SOURCES,
                  matching='Seed original Lyft/nuScenes assignment, then BEVFusion assignment with gates against every member',
                  method='score-weighted 2/3-source dimensions and score, strongest center/yaw, singleton retention',
                  axis_alignment='Groups containing BEVFusion: swap member DX/DY when local X is closer to strongest local Y; preserve original two-source baseline')
    fused, stages, provenance, per_frame = {}, Counter(), Counter(), []
    names = SOURCES+['kbf_original', 'two_model_fusion', 'three_model_fusion']
    totals, common_totals = {n: np.zeros(3, dtype=int) for n in names}, {n: np.zeros(3, dtype=int) for n in names}
    image_dir = output / 'comparison'
    image_dir.mkdir(exist_ok=True)
    for index, rows in enumerate(zip(*data)):
        frame_id = rows[0]['frame_id']
        # With two sources this implementation must reproduce the prior experiment.
        controlled, _ = fuse_frame(rows[:2], config, iou3d_nms_utils.boxes_bev_iou_cpu, nms)
        np.testing.assert_allclose(controlled['gt_boxes'], baseline[frame_id]['gt_boxes'], atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(controlled['source_indices'], baseline[frame_id]['source_indices'])
        result, stats = fuse_frame(rows, config, iou3d_nms_utils.boxes_bev_iou_cpu, nms)
        fused[frame_id] = result
        stages.update(stats)
        provenance.update(result['provenance'])
        boxes = {name: raw_boxes(row, config['output_score']) for name, row in zip(SOURCES, rows)}
        boxes.update(kbf_original=old_kbf[frame_id]['gt_boxes'], two_model_fusion=baseline[frame_id]['gt_boxes'],
                     three_model_fusion=result['gt_boxes'])
        record = dict(frame_id=frame_id, provenance=dict(Counter(result['provenance'])))
        for name, values in boxes.items():
            count = np.array([(values[:, 7] == c).sum() for c in [1, 2, 3]])
            selected = values[common_mask(values, extrinsic)]
            common = np.array([(selected[:, 7] == c).sum() for c in [1, 2, 3]])
            totals[name] += count
            common_totals[name] += common
            record[name] = count.tolist()
        per_frame.append(record)
        if not args.no_render:
            titles = ['Lyft', 'nuScenes OpenPCDet', 'BEVFusion LiDAR-only', 'KBF original (2 models)',
                      'Fusion 2 models + singletons', 'Fusion 3 models + singletons']
            panels = []
            for name, title in zip(names, titles):
                if name == 'three_model_fusion':
                    tags = ['singleton' if sum(r >= 0) == 1 else 'matched' for r in result['source_indices']]
                elif name == 'two_model_fusion':
                    tags = ['singleton' if 'singleton' in p else 'matched' for p in baseline[frame_id]['provenance']]
                else:
                    tags = ['original']*len(boxes[name])
                panels.append((title, boxes[name], tags))
            render_frame(frame_id, panels, ROOT / 'runs/two_model_detection/prepared', image_dir / f'{frame_id}.png')
        if (index+1) % 40 == 0 or index+1 == len(expected_ids):
            print(f'Fused/rendered {index+1}/{len(expected_ids)} frames', flush=True)
    with (output / 'fusion.pkl').open('wb') as stream:
        pickle.dump(fused, stream)
    summary = dict(frame_count=len(fused), classes=CLASSES, config=config,
                   totals={k: v.tolist() for k, v in totals.items()},
                   totals_common_range={k: v.tolist() for k, v in common_totals.items()},
                   common_range_definition='Box centers inside both BEVFusion native LiDAR range and OpenPCDet vehicle range',
                   stages=dict(stages), provenance=dict(provenance), per_frame=per_frame,
                   source_sha256=hashes, two_model_baseline_exactly_reproduced=True,
                   limitations=['No GT: output changes are not accuracy gains.',
                                'Fusion is source-order dependent: Lyft/nuScenes first, BEVFusion second.',
                                'Model scores are not calibrated.',
                                'BEVFusion transformed corners are represented as an upright enclosure.',
                                'Input feature/sweep counts and detector ranges differ.'])
    write_json(output / 'config.json', config)
    write_json(output / 'summary.json', summary)
    for path in input_paths:
        if fingerprint(path) != hashes[str(path.relative_to(ROOT))]:
            raise RuntimeError('Original artifact changed')
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_frame'}, indent=2))


if __name__ == '__main__':
    main()
