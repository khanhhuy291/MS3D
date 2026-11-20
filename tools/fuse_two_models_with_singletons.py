"""Cross-model pair fusion with explicit high-score singleton retention.

Reuses stored detections; writes to a separate experiment directory.
"""
import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from two_model_detection import CLASSES, MAPPING, write_json

DEFAULT_CONFIG = dict(
    admission_score=.1, output_score=.3, singleton_score=.7,
    match_bev_iou=[.3, .2, .2], match_center_xy_m=[2.0, .75, 1.0],
    match_center_z_m=1.0, final_nms_bev_iou=[.5, .3, .3],
    coordinate_frame='vehicle via T_vehicle_lidar.FM; no additional transform',
    method='one-to-one cross-model matching + intended KBF two-box rule + high-score singletons',
)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def select_pairs(ious, distance_xy, distance_z, min_iou, max_xy, max_z):
    """Maximum-cardinality gated assignment, with IoU resolving valid assignments."""
    from scipy.optimize import linear_sum_assignment
    if not ious.size:
        return np.empty((0, 2), dtype=int)
    valid = (ious >= min_iou) & (distance_xy <= max_xy) & (distance_z <= max_z)
    cost = np.where(valid, -(1.0 + min(ious.shape) + ious), 1e6)
    row, col = linear_sum_assignment(cost)
    return np.column_stack([row, col])[valid[row, col]].astype(int)


def fuse_pair(a, b):
    """Explicit two-box rule, avoiding the repository's tuple-shape branch bug.

    Rows: x,y,z,dx,dy,dz,yaw,class_id,score. Keep center/yaw/class of
    the stronger source, score-weighted mean of dimensions and scores.
    """
    if a[7] != b[7] or min(a[8], b[8]) <= 0:
        raise ValueError('Pairs must have the same supercategory and positive scores')
    pair = np.stack([a, b]).astype(np.float64)
    weights = pair[:, 8]
    strongest = int(np.argmax(weights))
    result = pair[strongest].copy()
    result[3:6] = np.average(pair[:, 3:6], axis=0, weights=weights)
    result[8] = np.average(pair[:, 8], weights=weights)
    return result.astype(np.float32)


def select_unmatched(boxes, matched_indices, threshold):
    used = set(matched_indices)
    return [i for i, box in enumerate(boxes) if i not in used and box[8] >= threshold]


def rows_for_class(row, class_id, admission):
    classes = np.array([MAPPING[name] for name in row['name']])
    ids = np.flatnonzero((classes == class_id) & (row['score'] >= admission))
    result = np.column_stack([row['boxes_lidar'][ids, :7], classes[ids], row['score'][ids]])
    return result, ids


def fuse_frame(rows, config, bev_iou, nms_fn):
    boxes_out, provenance_out, refs_out = [], [], []
    stages = Counter()
    for class_id in [1, 2, 3]:
        a, a_ids = rows_for_class(rows[0], class_id, config['admission_score'])
        b, b_ids = rows_for_class(rows[1], class_id, config['admission_score'])
        stages['lyft_admitted'] += len(a)
        stages['nuscenes_admitted'] += len(b)
        if len(a) and len(b):
            ious = bev_iou(a[:, :7].astype(np.float32), b[:, :7].astype(np.float32))
            xy = np.linalg.norm(a[:, None, :2]-b[None, :, :2], axis=2)
            z = abs(a[:, None, 2]-b[None, :, 2])
            pairs = select_pairs(ious, xy, z, config['match_bev_iou'][class_id-1],
                                 config['match_center_xy_m'][class_id-1], config['match_center_z_m'])
        else:
            pairs = np.empty((0, 2), dtype=int)
        stages['matched_pairs'] += len(pairs)
        candidates, provenance, refs = [], [], []
        for i, j in pairs:
            result = fuse_pair(a[i], b[j])
            if result[8] >= config['output_score']:
                candidates.append(result)
                provenance.append('matched')
                refs.append([int(a_ids[i]), int(b_ids[j])])
            else:
                stages['matched_below_output_score'] += 1
        for source, source_boxes, original_ids, used, column in [
            ('lyft_singleton', a, a_ids, pairs[:, 0], 0),
            ('nuscenes_singleton', b, b_ids, pairs[:, 1], 1),
        ]:
            selected = select_unmatched(source_boxes, used,
                                        max(config['singleton_score'], config['output_score']))
            stages[source+'_before_nms'] += len(selected)
            stages[source+'_below_threshold'] += len(source_boxes)-len(used)-len(selected)
            for i in selected:
                candidates.append(source_boxes[i].astype(np.float32))
                provenance.append(source)
                ref = [-1, -1]
                ref[column] = int(original_ids[i])
                refs.append(ref)
        candidates = np.array(candidates, dtype=np.float32).reshape(-1, 9)
        keep = (np.flatnonzero(nms_fn(candidates[:, :7], candidates[:, 8],
                                     config['final_nms_bev_iou'][class_id-1]))
                if len(candidates) > 1 else np.arange(len(candidates)))
        stages['candidates_before_nms'] += len(candidates)
        stages['removed_by_nms'] += len(candidates)-len(keep)
        for i in keep:
            boxes_out.append(candidates[i])
            provenance_out.append(provenance[i])
            refs_out.append(refs[i])
            stages[provenance[i]+'_after_nms'] += 1
    return dict(gt_boxes=np.array(boxes_out, dtype=np.float32).reshape(-1, 9),
                provenance=provenance_out,
                source_indices=np.array(refs_out, dtype=int).reshape(-1, 2)), stages


def render(args, results, baseline, fused, metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    image_dir = args.output / 'comparison'
    image_dir.mkdir(exist_ok=True)
    colors = np.array(['#00d2ff', '#ffcf33', '#ff66bb'])
    for index, (row_a, row_b) in enumerate(zip(results['lyft'], results['nuscenes'])):
        frame_id = row_a['frame_id']
        panels = []
        for name, row in [('Lyft', row_a), ('nuScenes', row_b)]:
            mask = row['score'] >= args.output_score
            labels = np.array([MAPPING[n] for n in row['name'][mask]])
            boxes = np.column_stack([row['boxes_lidar'][mask, :7], labels, row['score'][mask]])
            panels.append((name, boxes, ['original']*len(boxes)))
        panels.append(('KBF original', baseline[frame_id]['gt_boxes'],
                       ['original']*len(baseline[frame_id]['gt_boxes'])))
        panels.append(('Pair fusion + singletons', fused[frame_id]['gt_boxes'], fused[frame_id]['provenance']))
        xyz = np.load(args.input / 'prepared' / f'{frame_id}.npy')
        xyz = xyz[(abs(xyz[:, 0]) < args.radius) & (abs(xyz[:, 1]) < args.radius)][::2]
        fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
        for ax, (name, boxes, provenance) in zip(axes, panels):
            ax.set_facecolor('#101822')
            ax.scatter(xyz[:, 0], xyz[:, 1], s=.2, c='#8998a8', alpha=.6, rasterized=True)
            segments, segment_colors, styles = [], [], []
            for box, source in zip(boxes, provenance):
                x, y, z, dx, dy, dz, yaw, class_id, score = box
                corners = np.array([[dx/2, dy/2], [dx/2, -dy/2], [-dx/2, -dy/2],
                                    [-dx/2, dy/2], [dx/2, dy/2]])
                rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                corners = corners @ rot.T + [x, y]
                segments.extend([corners, np.array([[x, y], [x+dx/2*np.cos(yaw), y+dx/2*np.sin(yaw)]])])
                segment_colors.extend([colors[int(class_id)-1]]*2)
                styles.extend(['dashed' if 'singleton' in source else 'solid']*2)
            ax.add_collection(LineCollection(segments, colors=segment_colors, linewidths=.8, linestyles=styles))
            extra = (f' | paired {provenance.count("matched")} + single {len(boxes)-provenance.count("matched")}'
                     if name.startswith('Pair') else '')
            ax.set(xlim=(-args.radius, args.radius), ylim=(-args.radius, args.radius),
                   xlabel='vehicle X forward (m)', title=f'{name}: {len(boxes)}{extra}')
            ax.set_aspect('equal')
        axes[0].set_ylabel('vehicle Y left (m)')
        fig.suptitle(f'{frame_id} | output score >= {args.output_score} | vehicle frame | '
                     f'new: dashed = singleton score >= {args.singleton_score}; solid = pair')
        fig.tight_layout()
        fig.savefig(image_dir / f'{frame_id}.png', dpi=110)
        plt.close(fig)
        if (index+1) % 40 == 0 or index+1 == len(fused):
            print(f'Rendered {index+1}/{len(fused)} frames', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'runs/two_model_detection')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/two_model_detection_singletons')
    parser.add_argument('--admission', type=float, default=.1)
    parser.add_argument('--output-score', type=float, default=.3)
    parser.add_argument('--singleton-score', type=float, default=.7)
    parser.add_argument('--radius', type=float, default=40)
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    args.input, args.output = args.input.resolve(), args.output.resolve()
    if args.input == args.output:
        parser.error('Use a separate output directory to preserve the original run')
    if not (0 <= args.admission <= args.output_score <= args.singleton_score <= 1):
        parser.error('Require 0 <= admission <= output-score <= singleton-score <= 1')
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    from pcdet.utils.box_fusion_utils import nms

    config = dict(DEFAULT_CONFIG, admission_score=args.admission, output_score=args.output_score,
                  singleton_score=args.singleton_score)
    files = [args.input / name / 'result.pkl' for name in ['lyft', 'nuscenes']]
    files += [args.input / 'fusion/kbf.pkl', args.input / 'manifest.json']
    before = {str(path.relative_to(ROOT)): fingerprint(path) for path in files}
    results = {}
    for name, path in zip(['lyft', 'nuscenes'], files[:2]):
        with path.open('rb') as stream:
            results[name] = pickle.load(stream)
    with files[2].open('rb') as stream:
        baseline = pickle.load(stream)
    expected = [f['frame_id'] for f in json.loads(files[3].read_text())['frames']]
    for name, rows in results.items():
        if [row['frame_id'] for row in rows] != expected:
            raise ValueError(f'Input frame coverage/order mismatch: {name}')
    if list(baseline) != expected:
        raise ValueError('Original KBF frame coverage/order mismatch')
    fused, stages, provenance = {}, Counter(), Counter()
    totals = {name: np.zeros(3, dtype=int) for name in ['lyft', 'nuscenes', 'kbf_original', 'pair_singletons']}
    per_frame = []
    for row_a, row_b in zip(results['lyft'], results['nuscenes']):
        frame_id = row_a['frame_id']
        result, counters = fuse_frame([row_a, row_b], config, iou3d_nms_utils.boxes_bev_iou_cpu, nms)
        fused[frame_id] = result
        stages.update(counters)
        provenance.update(result['provenance'])
        count = dict(frame_id=frame_id, provenance=dict(Counter(result['provenance'])))
        for name, row in [('lyft', row_a), ('nuscenes', row_b)]:
            labels = np.array([MAPPING[n] for n in row['name'][row['score'] >= args.output_score]])
            count[name] = [int((labels == c).sum()) for c in [1, 2, 3]]
        for name, boxes in [('kbf_original', baseline[frame_id]['gt_boxes']), ('pair_singletons', result['gt_boxes'])]:
            count[name] = [int((boxes[:, 7] == c).sum()) for c in [1, 2, 3]]
        for name in totals:
            totals[name] += count[name]
        per_frame.append(count)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'fusion.pkl').open('wb') as stream:
        pickle.dump(fused, stream)
    write_json(args.output / 'config.json', config)
    summary = dict(frame_count=len(fused), classes=CLASSES, config=config,
                   totals={name: total.tolist() for name, total in totals.items()},
                   stages=dict(stages), provenance=dict(provenance), per_frame=per_frame,
                   source_sha256=before,
                   limitations=['No GT: counts and restored coverage do not measure accuracy.',
                                'This is a source-aware two-box fusion variant, not original MS3D KDE grouping.',
                                'Model scores are not calibrated to this dataset.',
                                'Plot range is +/-40m by default; counts cover the full detector range.'])
    write_json(args.output / 'summary.json', summary)
    if not args.no_render:
        render(args, results, baseline, fused, summary)
    after = {str(path.relative_to(ROOT)): fingerprint(path) for path in files}
    if before != after:
        raise RuntimeError('An original input artifact changed during the experiment')
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_frame'}, indent=2))


if __name__ == '__main__':
    main()
