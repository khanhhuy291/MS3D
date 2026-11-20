"""Run native LiDAR-top BEVFusion and export corner-validated vehicle-frame boxes."""
import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from two_model_detection import read_pcd, write_json
from fuse_two_models_with_singletons import fingerprint
from bevfusion_box_adapter import corners_to_vehicle_boxes


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bev-repo', type=Path, default=Path('/workspace/bevfusion-adapter'))
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/three_model_detection/bevfusion')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.bev_repo))
    import torch
    from mmdet3d.core import LiDARInstance3DBoxes
    from vf6_adapter.pcd import prepare_model_points

    spec = importlib.util.spec_from_file_location('bev_existing_inference', args.bev_repo / 'tools/infer_pcd.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    config_path = args.bev_repo / 'configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml'
    cfg = helper.load_config(str(config_path))
    checkpoint_path = ROOT / 'checkpoints/lidar-only-det.pth'
    model = helper.build_detector(cfg, str(checkpoint_path), 'cuda:0')
    disk = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['state_dict']
    expected = model.state_dict()
    if set(disk) != set(expected) or any(disk[k].shape != expected[k].shape for k in expected):
        raise RuntimeError('BEVFusion checkpoint key/shape mismatch')
    manifest = json.loads((ROOT / 'runs/two_model_detection/manifest.json').read_text())
    extrinsic = np.array(manifest['extrinsic'])
    # Exercise the real library corner convention before touching predictions.
    synthetic = LiDARInstance3DBoxes(torch.tensor([[1., 2., -2., 2., 4., 1.6, .6]]), box_dim=7)
    converted, _ = corners_to_vehicle_boxes(synthetic.corners.numpy(), np.eye(4))
    np.testing.assert_allclose(converted[0], [1, 2, -1.2, 2, 4, 1.6, -.6], atol=1e-5)
    classes, limits = list(cfg.object_classes), np.array(cfg.point_cloud_range)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'resolved_config.py').write_text(cfg.pretty_text)
    checkpoint_sha = fingerprint(checkpoint_path)
    records, native_records, times = [], [], []
    metadata = dict(checkpoint=str(checkpoint_path.relative_to(ROOT)), checkpoint_sha256=checkpoint_sha,
                    config=str(config_path), weights_loaded=len(disk), weights_expected=len(expected),
                    input_frame='raw LiDAR-top FM, no input transform or Z offset',
                    input_features=['x', 'y', 'z', 'intensity', 'time_lag=0'],
                    single_sweep=True, intensity_scale=1.0, classes=classes,
                    point_cloud_range=limits.tolist(), input_score_threshold=.1,
                    output_frame='vehicle via full T_vehicle_lidar.FM on corners',
                    output_box_contract='center XYZ, DX DY DZ, yaw of projected 0->4 corner edge; upright enclosure',
                    calibration=extrinsic.tolist(), library_box_geometry_test='PASS',
                    excluded_from_fusion=['barrier', 'traffic_cone'],
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(),
                    limitation='A tilted cuboid becomes an upright enclosure after extrinsic; size can increase. Input is single-sweep rather than training-time multi-sweep.')
    write_json(args.output / 'metadata.json', metadata)
    frames = manifest['frames'][:args.limit] if args.limit else manifest['frames']
    torch.cuda.reset_peak_memory_stats()
    for index, frame in enumerate(frames):
        path = ROOT / frame['source']
        raw, timestamp = read_pcd(path)
        if abs(timestamp-frame['timestamp']) > 1e-6:
            raise ValueError('PCD/manifest timestamp mismatch')
        points = prepare_model_points(raw, ['x', 'y', 'z', 'intensity'],
                                      intensity_scale=1.0, point_cloud_range=limits)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result = model(points=[torch.from_numpy(points).cuda()],
                           metas=[dict(box_type_3d=LiDARInstance3DBoxes, token=frame['frame_id'], timestamp=timestamp)])[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        scores_all = result['scores_3d'].cpu().numpy()
        mask = scores_all >= .1
        boxes_obj = result['boxes_3d'][mask]
        scores = scores_all[mask]
        labels = result['labels_3d'].cpu().numpy()[mask]
        names = np.array(classes)[labels]
        corners = boxes_obj.corners.cpu().numpy() if len(scores) else np.empty((0, 8, 3))
        boxes, transformed_corners = corners_to_vehicle_boxes(corners, extrinsic)
        center_native = boxes_obj.gravity_center.cpu().numpy()
        expected_center = center_native @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        np.testing.assert_allclose(boxes[:, :3], expected_center, atol=3e-5, rtol=1e-5)
        relevant = ~np.isin(names, metadata['excluded_from_fusion'])
        records.append(dict(frame_id=frame['frame_id'], timestamp=timestamp,
                            boxes_lidar=boxes[relevant], name=names[relevant], score=scores[relevant],
                            native_indices=np.flatnonzero(relevant)))
        native_records.append(dict(frame_id=frame['frame_id'], timestamp=timestamp,
                                   boxes_bottom_legacy=boxes_obj.tensor.cpu().numpy(), gravity_center=center_native,
                                   corners_lidar=corners, corners_vehicle=transformed_corners,
                                   converted_boxes=boxes, score=scores, name=names))
        times.append(elapsed)
        for filename, content in [('result.pkl', records), ('native_geometry.pkl', native_records)]:
            with (args.output / filename).open('wb') as stream:
                pickle.dump(content, stream)
        print(f'BEVFusion {index+1}/{len(frames)} {frame["frame_id"]}: '
              f'{int(relevant.sum())} relevant/{len(scores)} native boxes >=0.1, {elapsed:.3f}s', flush=True)
    metadata.update(frame_count=len(records), forward_seconds=times,
                    peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                    first_frame_intensity_percentiles=np.percentile(read_pcd(ROOT / frames[0]['source'])[0][:, 3], [0, 50, 99, 100]).tolist())
    write_json(args.output / 'metadata.json', metadata)


if __name__ == '__main__':
    main()
