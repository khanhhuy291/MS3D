"""Reproducible FM detection experiment; raw inputs are never modified.

Run from the repository root. Commands: prepare, infer, compare.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLASSES = ['Vehicle', 'Pedestrian', 'Cyclist']
MAPPING = {'car': 1, 'truck': 1, 'bus': 1, 'pedestrian': 2,
           'motorcycle': 3, 'bicycle': 3}
MODEL_FILES = {
    'lyft': ('lyft_models/uda_pv_rcnn_plusplus_resnet_anchorhead.yaml',
             'lyft_uda_pv_rcnn_plusplus_resnet_anchorhead_1xyz_allcls.pth'),
    'nuscenes': ('nuscenes_models/uda_voxel_rcnn_centerhead.yaml',
                 'nuscenes_uda_voxel_rcnn_centerhead_10xyzt_allcls.pth'),
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def read_pcd(path):
    """Read the verified binary float32 XYZI contract, rejecting other formats."""
    header, timestamp = {}, None
    with path.open('rb') as stream:
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f'Missing PCD DATA header: {path}')
            line = raw.decode('ascii').strip()
            if line.startswith('# timestamp '):
                timestamp = float(line.split()[-1])
            if not line or line.startswith('#'):
                continue
            key, *values = line.split()
            header[key] = values
            if key == 'DATA':
                break
        if (header['DATA'] != ['binary'] or header['FIELDS'] != ['x', 'y', 'z', 'intensity']
                or header['SIZE'] != ['4'] * 4 or header['TYPE'] != ['F'] * 4
                or header['COUNT'] != ['1'] * 4):
            raise ValueError(f'Unsupported PCD layout: {path}')
        points = np.frombuffer(stream.read(), dtype='<f4').reshape(-1, 4).copy()
    if len(points) != int(header['POINTS'][0]) or timestamp is None:
        raise ValueError(f'Invalid PCD count/timestamp: {path}')
    return points, timestamp


def prepare(args):
    import yaml
    from scipy.spatial.transform import Rotation, Slerp

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    extrinsic = np.array(json.loads((args.raw / 'calib_new2.json').read_text())
                         ['T_vehicle_lidar']['FM'], dtype=np.float64)
    np.testing.assert_allclose(extrinsic[3], [0, 0, 0, 1], atol=1e-8)
    np.testing.assert_allclose(extrinsic[:3, :3].T @ extrinsic[:3, :3], np.eye(3), atol=1e-6)
    trajectory = np.loadtxt(args.raw / 'trajectory_optimized.tum')
    if not np.all(np.diff(trajectory[:, 0]) > 0):
        raise ValueError('Trajectory timestamps must be strictly increasing')
    rotations = Slerp(trajectory[:, 0], Rotation.from_quat(trajectory[:, 4:8]))
    files = sorted((args.raw / '20260916_134800_lidar').glob('*.pcd'))
    cache = out / 'prepared'
    cache.mkdir(exist_ok=True)
    frames, finite_dropped, max_roundtrip_error = [], 0, 0.0
    for path in files:
        points, timestamp = read_pcd(path)
        if not trajectory[0, 0] <= timestamp <= trajectory[-1, 0]:
            raise ValueError(f'Pose extrapolation forbidden: {path}')
        good = np.isfinite(points[:, :3]).all(axis=1)
        finite_dropped += int((~good).sum())
        lidar = points[good, :3].astype(np.float64)
        vehicle = lidar @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        recovered = (vehicle - extrinsic[:3, 3]) @ extrinsic[:3, :3]
        max_roundtrip_error = max(max_roundtrip_error, float(np.max(abs(recovered-lidar))))
        world_from_lidar = np.eye(4)
        world_from_lidar[:3, :3] = rotations(timestamp).as_matrix()
        world_from_lidar[:3, 3] = [np.interp(timestamp, trajectory[:, 0], trajectory[:, k])
                                 for k in (1, 2, 3)]
        world_from_vehicle = world_from_lidar @ np.linalg.inv(extrinsic)
        np.save(cache / f'{path.stem}.npy', vehicle.astype(np.float32))
        frames.append(dict(frame_id=path.stem, timestamp=timestamp,
                           source=str(path.relative_to(ROOT)), num_points=len(vehicle),
                           pose=world_from_vehicle.tolist()))
    if not frames or not np.all(np.diff([f['timestamp'] for f in frames]) > 0):
        raise ValueError('PCD timestamps must be present and strictly increasing')
    manifest = dict(coordinate_frame='vehicle via T_vehicle_lidar.FM; no extra Z shift',
                    extrinsic=extrinsic.tolist(), frames=frames,
                    pose_interpolation='linear translation + quaternion SLERP',
                    finite_points_dropped=finite_dropped,
                    transform_roundtrip_max_abs_error_m=max_roundtrip_error)
    write_json(out / 'manifest.json', manifest)
    for name, (cfg_name, _) in MODEL_FILES.items():
        config = yaml.safe_load((ROOT / 'tools/cfgs' / cfg_name).read_text())
        features = ['x', 'y', 'z'] + (['timestamp'] if name == 'nuscenes' else [])
        voxel = dict(NAME='transform_points_to_voxels', VOXEL_SIZE=[0.1, 0.1, 0.15],
                     MAX_POINTS_PER_VOXEL=5, MAX_NUMBER_OF_VOXELS={'train': 80000, 'test': 90000})
        if name == 'nuscenes':
            voxel = dict(NAME='transform_points_to_voxels_placeholder', VOXEL_SIZE=[0.1, 0.1, 0.15])
        config['DATA_CONFIG'] = dict(
            DATA_PATH='.', POINT_CLOUD_RANGE=[-75.2, -75.2, -2, 75.2, 75.2, 4],
            SHIFT_COOR=[0.0, 0.0, 0.0], USE_TTA=False,
            POINT_FEATURE_ENCODING=dict(encoding_type='absolute_coordinates_encoding',
                                        used_feature_list=features, src_feature_list=features),
            DATA_PROCESSOR=[dict(NAME='mask_points_and_boxes_outside_range', REMOVE_OUTSIDE_BOXES=True),
                            dict(NAME='shuffle_points', SHUFFLE_ENABLED={'train': True, 'test': False}), voxel])
        config['MODEL']['POST_PROCESSING']['SCORE_THRESH'] = 0.1
        config_dir = out / 'configs'
        config_dir.mkdir(exist_ok=True)
        (config_dir / f'{name}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'frames'}, indent=2))
    print(f'Prepared {len(frames)} frames and two independent model configs in {out}')


def infer(args):
    import torch
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    common_utils.set_random_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((args.output / 'manifest.json').read_text())
    frames = manifest['frames']
    selected = list(range(args.start, len(frames), args.stride))
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError('No frames selected; check --start and --limit')
    config = EasyDict()
    cfg_from_yaml_file(str(args.output / 'configs' / f'{args.model}.yaml'), config)
    sweeps = 1 if args.model == 'lyft' else args.sweeps
    logger = common_utils.create_logger()

    class PreparedDataset(DatasetTemplate):
        def __init__(self):
            super().__init__(config.DATA_CONFIG, config.CLASS_NAMES, training=False,
                             root_path=ROOT, logger=logger)

        def __len__(self):
            return len(selected)

        def __getitem__(self, item):
            index = selected[item]
            current = frames[index]
            to_current = np.linalg.inv(np.array(current['pose']))
            parts = []
            # Use only available history, never duplicate the first frame.
            for history in range(index, max(-1, index-sweeps), -1):
                frame = frames[history]
                xyz = np.load(args.output / 'prepared' / f"{frame['frame_id']}.npy")
                transform = to_current @ np.array(frame['pose'])
                xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
                # Suppress roof/ego returns in the VEHICLE frame.
                xyz = xyz[~((abs(xyz[:, 0]) < 1.5) & (abs(xyz[:, 1]) < 1.5))]
                if args.model == 'nuscenes':
                    lag = np.full((len(xyz), 1), current['timestamp']-frame['timestamp'])
                    xyz = np.hstack([xyz, lag])
                parts.append(xyz)
            points = np.concatenate(parts).astype(np.float32)
            return self.prepare_data(dict(points=points, frame_id=current['frame_id']))

    dataset = PreparedDataset()
    model = build_network(config.MODEL, len(config.CLASS_NAMES), dataset)
    checkpoint = ROOT / 'checkpoints' / MODEL_FILES[args.model][1]
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state']
    expected, loaded = model._load_state_dict(state, strict=False)
    missing = sorted(set(expected)-set(loaded))
    unexpected = sorted(set(state)-set(expected))
    metadata = dict(model=args.model, checkpoint=str(checkpoint.relative_to(ROOT)),
                    weights_loaded=len(loaded), weights_expected=len(expected),
                    missing_keys=missing, unexpected_keys=unexpected,
                    max_sweeps=sweeps,
                    max_history_seconds=max(frames[i]['timestamp']-frames[max(0, i-sweeps+1)]['timestamp']
                                            for i in selected),
                    features=config.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list,
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), coordinate_frame=manifest['coordinate_frame'])
    dest = args.output / args.model
    write_json(dest / 'metadata.json', metadata)
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint mismatch; see {dest / "metadata.json"}')
    model.cuda().eval()
    results, timings = [], []
    torch.cuda.reset_peak_memory_stats()
    for item, index in enumerate(selected):
        batch = dataset.collate_batch([dataset[item]])
        load_data_to_gpu(batch)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            predictions, _ = model(batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        prediction = predictions[0]
        labels = prediction['pred_labels'].cpu().numpy()
        boxes = prediction['pred_boxes'].cpu().numpy()[:, :7]
        scores = prediction['pred_scores'].cpu().numpy()
        if (not np.isfinite(boxes).all() or not np.isfinite(scores).all()
                or not (boxes[:, 3:6] > 0).all()):
            raise RuntimeError(f'Invalid detector output at {frames[index]["frame_id"]}')
        results.append(dict(frame_id=frames[index]['frame_id'],
                            timestamp=frames[index]['timestamp'],
                            boxes_lidar=boxes, score=scores,
                            name=np.array(config.CLASS_NAMES)[labels-1]))
        timings.append(elapsed)
        # Persist each successful frame so interrupted runs remain inspectable.
        with (dest / 'result.pkl').open('wb') as stream:
            pickle.dump(results, stream)
        print(f'{args.model} {item+1}/{len(selected)} {frames[index]["frame_id"]}: '
              f'{len(labels)} boxes, forward {elapsed:.3f}s', flush=True)
    metadata.update(frame_count=len(results), forward_seconds=timings,
                    peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20)
    write_json(dest / 'metadata.json', metadata)


def compare(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pcdet.utils import box_fusion_utils as fusion

    results = {}
    for name in MODEL_FILES:
        with (args.output / name / 'result.pkl').open('rb') as stream:
            results[name] = pickle.load(stream)
    ids = [[row['frame_id'] for row in rows] for rows in results.values()]
    if ids[0] != ids[1]:
        raise ValueError('Model frame IDs or order differ')
    by_frame = {f['frame_id']: f for f in json.loads((args.output / 'manifest.json').read_text())['frames']}
    config = {c: dict(cls_id=i+1, discard=2, radius=[1.5, .3, .2][i],
                      nms=[.1, .3, .1][i], neg_th=args.threshold)
              for i, c in enumerate(CLASSES)}
    # Use the repository KBF entry point, without tracking or temporal refinement.
    sources = {name: rows for name, rows in results.items()}
    sources['det_cls_weights'] = {name: np.ones(3, dtype=int) for name in results}
    detection_sets = fusion.get_detection_sets(sources, score_th=args.threshold)
    fused = fusion.get_initial_pseudo_labels(detection_sets, config)
    fusion_dir = args.output / 'fusion'
    fusion_dir.mkdir(exist_ok=True)
    with (fusion_dir / 'kbf.pkl').open('wb') as stream:
        pickle.dump(fused, stream)
    image_dir = args.output / 'comparison'
    image_dir.mkdir(exist_ok=True)
    counts, totals = [], {name: [0, 0, 0] for name in ['lyft', 'nuscenes', 'kbf']}
    support_totals = dict(both_models=0, lyft_only=0, nuscenes_only=0, neither=0)
    colors = ['#00d2ff', '#ffcf33', '#ff66bb']
    for index, frame_id in enumerate(ids[0]):
        panels = []
        for name in MODEL_FILES:
            row = results[name][index]
            mask = row['score'] >= args.threshold
            panels.append((name, row['boxes_lidar'][mask],
                           np.array([MAPPING[n] for n in row['name'][mask]]), row['score'][mask]))
        boxes = fused[frame_id]['gt_boxes']
        panels.append(('kbf', boxes[:, :7], boxes[:, 7].astype(int), boxes[:, 8]))
        record = dict(frame_id=frame_id)
        support = dict.fromkeys(support_totals, 0)
        for box in boxes:
            class_id = int(box[7])
            nearby = []
            for _, source_boxes, source_labels, _ in panels[:2]:
                candidates = source_boxes[source_labels == class_id]
                nearby.append(bool(len(candidates) and np.any(
                    np.linalg.norm(candidates[:, :3]-box[:3], axis=1)
                    <= config[CLASSES[class_id-1]]['radius'])))
            support_key = ('both_models' if all(nearby) else 'lyft_only' if nearby[0]
                           else 'nuscenes_only' if nearby[1] else 'neither')
            support[support_key] += 1
            support_totals[support_key] += 1
        record['kbf_source_support'] = support
        for name, _, labels, _ in panels:
            record[name] = [int((labels == k).sum()) for k in [1, 2, 3]]
            totals[name] = (np.array(totals[name])+record[name]).tolist()
        counts.append(record)
        xyz = np.load(args.output / 'prepared' / f'{frame_id}.npy')
        roi = (abs(xyz[:, 0]) < args.radius) & (abs(xyz[:, 1]) < args.radius)
        xyz = xyz[roi][::2]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        for ax, (name, boxes, labels, scores) in zip(axes, panels):
            ax.set_facecolor('#101822')
            ax.scatter(xyz[:, 0], xyz[:, 1], s=.2, c='#8998a8', alpha=.6, rasterized=True)
            for box, label, score in zip(boxes, labels, scores):
                x, y, z, dx, dy, dz, yaw = box
                corners = np.array([[dx/2, dy/2], [dx/2, -dy/2], [-dx/2, -dy/2],
                                    [-dx/2, dy/2], [dx/2, dy/2]])
                rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                corners = corners @ rotation.T + [x, y]
                color = colors[label-1]
                ax.plot(corners[:, 0], corners[:, 1], c=color, lw=.8)
                ax.plot([x, x+dx/2*np.cos(yaw)], [y, y+dx/2*np.sin(yaw)], c=color, lw=.8)
            ax.set(xlim=(-args.radius, args.radius), ylim=(-args.radius, args.radius),
                   xlabel='vehicle X forward (m)', title=f'{name}: {len(boxes)} boxes')
            ax.set_aspect('equal')
        axes[0].set_ylabel('vehicle Y left (m)')
        fig.suptitle(f'{frame_id} | score >= {args.threshold} | vehicle frame | '
                     'cyan: vehicle, yellow: pedestrian, pink: cyclist')
        fig.tight_layout()
        fig.savefig(image_dir / f'{frame_id}.png', dpi=110)
        plt.close(fig)
        if (index+1) % 20 == 0 or index+1 == len(ids[0]):
            print(f'Rendered {index+1}/{len(ids[0])} comparison frames', flush=True)
    summary = dict(frame_count=len(ids[0]), classes=CLASSES, totals=totals, per_frame=counts,
                   score_threshold=args.threshold, kbf=config,
                   kbf_source_support=support_totals,
                   support_definition='Same supercategory and 3D center distance <= class KBF radius; not IoU or ground truth.',
                   limitations=['No ground truth: counts are not accuracy.',
                                'KBF discard=2 counts nearby boxes, not distinct models.',
                                'KBF can remove detections supported by only one box.',
                                'Images show +/- radius; counts cover the full model range.'])
    write_json(args.output / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_frame'}, indent=2))


def verify(args):
    """Check completed artifact coverage, validity and independently recounted totals."""
    out = args.output
    frames = json.loads((out / 'manifest.json').read_text())['frames']
    expected_ids = [f['frame_id'] for f in frames]
    summary = json.loads((out / 'summary.json').read_text())
    if summary['frame_count'] != len(frames):
        raise ValueError('Summary does not cover the full sequence')
    totals = {}
    for name in MODEL_FILES:
        with (out / name / 'result.pkl').open('rb') as stream:
            rows = pickle.load(stream)
        if [row['frame_id'] for row in rows] != expected_ids:
            raise ValueError(f'Incomplete or differently ordered model frames: {name}')
        metadata = json.loads((out / name / 'metadata.json').read_text())
        if (metadata['frame_count'] != len(frames) or metadata['missing_keys']
                or metadata['unexpected_keys'] or metadata['weights_loaded'] != metadata['weights_expected']):
            raise ValueError(f'Incomplete checkpoint or run: {name}')
        counts = np.zeros(3, dtype=int)
        for row in rows:
            boxes, scores = row['boxes_lidar'], row['score']
            if not np.isfinite(boxes).all() or not np.isfinite(scores).all() or not (boxes[:, 3:6] > 0).all():
                raise ValueError(f'Invalid box values: {name}/{row["frame_id"]}')
            labels = np.array([MAPPING[n] for n in row['name'][scores >= summary['score_threshold']]])
            counts += [(labels == i).sum() for i in [1, 2, 3]]
        totals[name] = counts.tolist()
    with (out / 'fusion/kbf.pkl').open('rb') as stream:
        fused = pickle.load(stream)
    if list(fused) != expected_ids:
        raise ValueError('Fusion frame IDs differ from input')
    counts = np.zeros(3, dtype=int)
    for frame_id, row in fused.items():
        boxes = row['gt_boxes']
        if boxes.ndim != 2 or boxes.shape[1] != 9 or not np.isfinite(boxes).all() or not (boxes[:, 3:6] > 0).all():
            raise ValueError(f'Invalid fusion boxes: {frame_id}')
        counts += [(boxes[:, 7] == i).sum() for i in [1, 2, 3]]
        image_path = out / 'comparison' / f'{frame_id}.png'
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise ValueError(f'Missing comparison image: {frame_id}')
    totals['kbf'] = counts.tolist()
    if totals != summary['totals']:
        raise ValueError('Independent totals disagree with summary')
    report = dict(status='PASS', frame_count=len(frames), independently_counted_totals=totals,
                  checks=['Frame ID coverage and order', 'Full checkpoint loading',
                          'Finite boxes and positive dimensions', 'Comparison image coverage',
                          'Independent count consistency'])
    write_json(out / 'verification.json', report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['prepare', 'infer', 'compare', 'verify'])
    parser.add_argument('--raw', type=Path, default=ROOT / 'data_raw')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/two_model_detection')
    parser.add_argument('--model', choices=list(MODEL_FILES), default='lyft')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--sweeps', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=.3)
    parser.add_argument('--radius', type=float, default=40)
    args = parser.parse_args()
    if args.sweeps < 1 or args.stride < 1 or args.start < 0 or args.limit < 0:
        parser.error('Invalid sweeps, stride, start or limit')
    args.output = args.output.resolve()
    args.raw = args.raw.resolve()
    {'prepare': prepare, 'infer': infer, 'compare': compare, 'verify': verify}[args.command](args)


if __name__ == '__main__':
    main()
