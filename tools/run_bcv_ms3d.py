"""Fuse VoxelRCNN + CenterPoint + BEVFusion and run isolated MS3D++ round1.

Existing detector outputs are reused. CenterPoint is transformed with the same
validated adapter as the viewer. Native vehicle/pedestrian refinement is kept;
cyclist detections are separately preserved with spatial/score filtering.
"""
import argparse
import importlib.util
import json
import pickle
from collections import Counter

import numpy as np

import run_ms3d_from_three_model_fusion as bridge
from fuse_three_models import DEFAULT_CONFIG, MAPPING3, fuse_frame
from fuse_two_models_with_singletons import fingerprint
from two_model_detection import write_json

ROOT = bridge.ROOT
FUSION = ROOT / 'runs/fusion_bcv'
REFINED = ROOT / 'runs/ms3d_bcv_round1'
SOURCE_ORDER = ['voxelrcnn', 'centerpoint', 'bevfusion']
CP = ROOT / 'runs/centerpoint_voxel1440_fm_singleframe'
SOURCE_PATHS = [ROOT / 'runs/two_model_detection/nuscenes/result.pkl',
                FUSION / 'centerpoint_vehicle.pkl',
                ROOT / 'runs/three_model_detection/bevfusion/result.pkl']


def configure_bridge():
    bridge.OUT = REFINED
    bridge.SOURCE = FUSION / 'fusion.pkl'
    bridge.PRESERVE_CYCLISTS = True


def prepare_and_fuse():
    from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu
    from pcdet.utils.box_fusion_utils import nms

    manifest = json.loads(bridge.MANIFEST.read_text())
    ids = [f['frame_id'] for f in manifest['frames']]
    raw_paths = [SOURCE_PATHS[0], SOURCE_PATHS[2], bridge.MANIFEST, CP / 'summary.json']
    raw_paths += sorted((CP / 'predictions').glob('*.json'))
    hashes = {str(p.relative_to(ROOT)): fingerprint(p) for p in raw_paths}
    FUSION.mkdir(parents=True, exist_ok=True)
    baseline_paths = [ROOT / 'runs/three_model_detection/fusion.pkl',
                      ROOT / 'runs/ms3d_three_model_round1/final_ps_dict.pkl']
    baseline_hashes = {str(p.relative_to(ROOT)): fingerprint(p) for p in baseline_paths}
    baseline_file = FUSION / 'baseline_sha256.json'
    if baseline_file.exists():
        assert json.loads(baseline_file.read_text()) == baseline_hashes
    else:
        write_json(baseline_file, baseline_hashes)
    config = dict(DEFAULT_CONFIG, source_order=SOURCE_ORDER, align_all_pairs=True,
                  matching='Seed VoxelRCNN/CenterPoint one-to-one, then BEVFusion gated against every member',
                  method='Strongest center/yaw; score-weighted axis-aligned dimensions/score; singleton retention')
    summary_path = FUSION / 'summary.json'
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing['source_sha256'] != hashes or existing['config'] != config:
            raise RuntimeError('BCV output already exists with different inputs/config')
        print('Using existing identical BCV fusion; verification still required.', flush=True)
        return
    # Reuse the already verified JSON/frame/transform adapter, without loading
    # viewer cases or attaching an HTTP server.
    spec = importlib.util.spec_from_file_location('bcv_viewer_adapter', ROOT / 'web/local_viewer/server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = module.ViewerStore.__new__(module.ViewerStore)
    adapter.manifest, adapter.frames = manifest, manifest['frames']
    cp_rows = adapter.load_centerpoint(CP / 'summary.json')
    excluded = Counter()
    for row in cp_rows:
        names = np.array(row['name'])
        mask = np.isin(names, list(MAPPING3))
        excluded.update(names[~mask].tolist())
        row['original_indices'] = np.flatnonzero(mask)
        row['name'], row['boxes_lidar'], row['score'] = names[mask], row['boxes_lidar'][mask].astype(np.float32), row['score'][mask].astype(np.float32)
    bridge.save(SOURCE_PATHS[1], cp_rows)
    data = [bridge.load(p) for p in SOURCE_PATHS]
    for rows in data:
        assert [r['frame_id'] for r in rows] == ids
        for row in rows:
            assert np.isfinite(row['boxes_lidar']).all() and np.isfinite(row['score']).all()
            assert (row['boxes_lidar'][:, 3:6] > 0).all()
    fused, stats, provenance, per_frame = {}, Counter(), Counter(), []
    raw_totals = {name: np.zeros(3, dtype=int) for name in SOURCE_ORDER}
    for index, rows in enumerate(zip(*data)):
        fid = ids[index]
        record, stage = fuse_frame(rows, config, boxes_bev_iou_cpu, nms)
        fused[fid] = record
        stats.update(stage)
        provenance.update(record['provenance'])
        per_frame.append(dict(frame_id=fid, counts=[int((record['gt_boxes'][:, 7] == c).sum()) for c in [1,2,3]],
                              provenance=dict(Counter(record['provenance']))))
        for name, row in zip(SOURCE_ORDER, rows):
            labels = np.array([MAPPING3[n] for n in row['name']])
            raw_totals[name] += [int(((labels == c) & (row['score'] >= .3)).sum()) for c in [1,2,3]]
        if (index+1)%40 == 0 or index+1 == len(ids):
            print(f'BCV fusion {index+1}/{len(ids)}', flush=True)
    bridge.save(FUSION / 'fusion.pkl', fused)
    summary = dict(frame_count=len(fused), source_order=SOURCE_ORDER, config=config,
                   totals={**{k:v.tolist() for k,v in raw_totals.items()},
                           'fusion': np.array([f['counts'] for f in per_frame]).sum(axis=0).tolist()},
                   stages=dict(stats), provenance=dict(provenance), per_frame=per_frame,
                   source_paths=[str(p.relative_to(ROOT)) for p in SOURCE_PATHS], source_sha256=hashes,
                   prepared_source_sha256={str(p.relative_to(ROOT)): fingerprint(p) for p in SOURCE_PATHS},
                   centerpoint_transform='model_lidar -> vehicle: Z+1.8m; yaw already length-axis',
                   excluded_centerpoint_classes=dict(excluded),
                   limitations=['No ground truth; counts are not accuracy.', 'Barrier/traffic_cone excluded from three-class MS3D fusion.',
                                'CenterPoint single sweep despite10-sweep training; original zero intensity retained.',
                                'Sequential matching is source-order dependent.'])
    write_json(summary_path, summary)
    print('BCV FUSION', json.dumps(summary['totals']), flush=True)
    for path, digest in hashes.items():
        assert fingerprint(ROOT / path) == digest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase', choices=['all','fusion','track','refine'], default='all')
    args = parser.parse_args()
    configure_bridge()
    if args.phase in ['all','fusion']:
        prepare_and_fuse()
    if args.phase != 'fusion':
        config = bridge.prepare()
        if args.phase in ['all','track']:
            bridge.track(config)
        if args.phase in ['all','refine']:
            bridge.refine(config)
        for path, digest in json.loads((REFINED / 'run_metadata.json').read_text())['source_sha256'].items():
            assert fingerprint(ROOT / path) == digest


if __name__ == '__main__':
    main()
