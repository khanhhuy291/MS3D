"""Run native MS3D++ tracking/refinement on the stored three-model fusion.

Only the dataset bridge, geometry-equivalent vehicle axis normalization and
read-only instrumentation are custom; the tracker/refinement algorithms are
the repository implementations. Stages are cached for resumable execution.
"""
import argparse
import copy
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tracker'))
from fuse_two_models_with_singletons import fingerprint
from two_model_detection import write_json

OUT = ROOT / 'runs/ms3d_three_model_round1'
SOURCE = ROOT / 'runs/three_model_detection/fusion.pkl'
MANIFEST = ROOT / 'runs/two_model_detection/manifest.json'
TEMPLATE = ROOT / 'tools/cfgs/target_custom/label_generation/round1/cfgs/ps_config.yaml'
PRESERVE_CYCLISTS = False


def load(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('wb') as stream:
        pickle.dump(value, stream)


def canonicalize_vehicle_boxes(boxes):
    result = np.array(boxes, copy=True)
    swap = (abs(result[:, 7]) == 1) & (result[:, 3] < result[:, 4])
    result[swap, 3:5] = result[swap, 3:5][:, ::-1]
    result[swap, 6] += np.pi/2
    result[:, 6] = (result[:, 6]+np.pi) % (2*np.pi)-np.pi
    return result, swap


class ManifestDataset:
    """The read/metadata contract consumed by compatibility_utils, no detector loader."""
    def __init__(self, manifest=MANIFEST):
        from easydict import EasyDict
        content = json.loads(Path(manifest).read_text())
        self.dataset_cfg = EasyDict(DATASET='CustomDataset', SHIFT_COOR=[0., 0., 0.])
        self.infos, self.frameid_to_idx = [], {}
        self.sequence = '20260916_134800_lidar'
        for index, frame in enumerate(content['frames']):
            self.frameid_to_idx[frame['frame_id']] = index
            self.infos.append(dict(frame_id=frame['frame_id'], timestamp=frame['timestamp'],
                                   pose=np.array(frame['pose']),
                                   lidar_path=str(ROOT / 'runs/two_model_detection/prepared' / f"{frame['frame_id']}.npy"),
                                   point_cloud=dict(lidar_sequence=self.sequence, sample_idx=index)))
        self.seq_name_to_len = {self.sequence: len(self.infos)}
        self.current_frame = None

    def get_lidar(self, path):
        self.current_frame = Path(path).stem
        return np.load(path, allow_pickle=False).astype(np.float32, copy=True)


def label_stats(labels):
    counts = Counter()
    for value in labels.values():
        boxes = value['gt_boxes']
        for cid, name in [(1, 'vehicle'), (2, 'pedestrian'), (3, 'cyclist')]:
            counts[name] += int((abs(boxes[:, 7]) == cid).sum())
            counts[name+'_positive'] += int((boxes[:, 7] == cid).sum())
            counts[name+'_ignored'] += int((boxes[:, 7] == -cid).sum())
    return dict(frame_count=len(labels), box_count=sum(len(x['gt_boxes']) for x in labels.values()), **counts)


def track_stats(tracks):
    lengths = [len(t['boxes']) for t in tracks.values()]
    result = dict(track_count=len(tracks), box_count=sum(lengths),
                  min_length=min(lengths, default=0), max_length=max(lengths, default=0),
                  median_length=float(np.median(lengths)) if lengths else 0,
                  non_monotonic_tracks=sum(any(a>b for a,b in zip(t['frame_id'], t['frame_id'][1:])) for t in tracks.values()))
    if tracks and all('motion_state' in t for t in tracks.values()):
        result['motion_state'] = dict(Counter('static' if int(np.median(t['motion_state'])) == 0 else 'dynamic' for t in tracks.values()))
    for key in ['frameid_to_box', 'frameid_to_rollingkde', 'frameid_to_propboxes']:
        if any(key in t for t in tracks.values()):
            result[key+'_count'] = sum(len(t.get(key, {})) for t in tracks.values())
    return result


def prepare():
    import yaml
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {str(p.relative_to(ROOT)): fingerprint(p) for p in [SOURCE, MANIFEST, TEMPLATE]}
    metadata_path = OUT / 'run_metadata.json'
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old['source_sha256'] != hashes:
            raise RuntimeError('Existing run uses different inputs/config; choose a new experiment directory')
        if old.get('preserve_cyclists', False) != PRESERVE_CYCLISTS:
            raise RuntimeError('Existing run uses a different cyclist output policy')
        return yaml.safe_load((OUT / 'ps_config.yaml').read_text())
    source = load(SOURCE)
    frames = json.loads(MANIFEST.read_text())['frames']
    if list(source) != [frame['frame_id'] for frame in frames]:
        raise ValueError('Fusion/manifest frame order mismatch')
    initial, swaps = {}, {}
    for frame_id, value in source.items():
        boxes, mask = canonicalize_vehicle_boxes(value['gt_boxes'])
        initial[frame_id] = dict(gt_boxes=boxes)
        if mask.any():
            swaps[frame_id] = np.flatnonzero(mask).tolist()
    config = yaml.safe_load(TEMPLATE.read_text())
    config.update(DETS_TXT=None, DATA_CONFIG_PATH=None, SAVE_DIR=str(OUT),
                  INPUT_FUSION=str(SOURCE), DATA_ADAPTER='ManifestDataset; prepared vehicle XYZ + manifest pose/timestamp',
                  COORDINATE_FRAME='vehicle; SHIFT_COOR=0; pose=world_from_vehicle',
                  FRAME_RATE_POLICY='Actual timestamps in seconds; native round1 frame thresholds unchanged')
    if PRESERVE_CYCLISTS:
        config['CYCLIST_OUTPUT_POLICY'] = 'Preserve fusion detections with NMS0.3, current-sweep points>1, ego filter and POS_TH; no temporal refinement'
    save(OUT / 'input_fusion3_original.pkl', source)
    save(OUT / 'initial_pseudo_labels.pkl', initial)
    (OUT / 'ps_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = dict(source_sha256=hashes, frame_count=len(frames), input_stats=label_stats(initial),
                    preserve_cyclists=PRESERVE_CYCLISTS,
                    singleton_threshold_in_source=.7, group_threshold_in_source=.3,
                    exact_singleton_scores_equal_0_7=sum(float(box[8]) == .7 for value in source.values()
                                                        for box, ref in zip(value['gt_boxes'], value['source_indices']) if (ref >= 0).sum() == 1),
                    vehicle_axis_swaps=sum(map(len, swaps.values())), swaps_by_frame=swaps,
                    timestamp_units='seconds', median_frame_dt=float(np.median(np.diff([f['timestamp'] for f in frames]))),
                    coordinate_frame='vehicle', z_contract='geometric center', shift_coor=[0., 0., 0.],
                    data_pose='T_world_vehicle from manifest = T_world_lidar @ inverse(T_vehicle_lidar.FM)',
                    provenance_policy='Original source_indices/provenance retained in input archive; not copied onto changed final boxes',
                    native_algorithms=['tracker_utils.get_tracklets', 'ms3d_utils.refine_veh_labels',
                                       'ms3d_utils.refine_ped_labels', 'ms3d_utils.update_ps', 'ms3d_utils.select_ps_by_th'])
    write_json(metadata_path, metadata)
    print('PREPARE', json.dumps(metadata['input_stats']), 'vehicle axis swaps', metadata['vehicle_axis_swaps'], flush=True)
    return config


def track(config):
    from pcdet.utils import tracker_utils
    dataset, labels = ManifestDataset(), load(OUT / 'initial_pseudo_labels.pkl')
    summaries = {}
    for name, cls_id, cfg_key in [('veh_all', 1, 'VEH_ALL'), ('veh_static', 1, 'VEH_STATIC'), ('ped', 2, 'PEDESTRIAN')]:
        destination = OUT / f'tracks_world_{name}.pkl'
        started = time.perf_counter()
        if destination.exists():
            tracks = load(destination)
        else:
            trk_config = tracker_utils.prepare_track_cfg(config['TRACKING'][cfg_key])
            tracks = tracker_utils.get_tracklets(dataset, labels, trk_config, cls_id)
            save(destination, tracks)
        summaries[name] = dict(**track_stats(tracks), elapsed_seconds=time.perf_counter()-started)
        write_json(OUT / 'tracking_summary.json', summaries)
        print('TRACK', name, json.dumps(summaries[name]), flush=True)


def refine(config):
    from pcdet.utils import ms3d_utils as native
    dataset = ManifestDataset()
    initial = load(OUT / 'initial_pseudo_labels.pkl')
    all_tracks, static_tracks, ped_tracks = [load(OUT / f'tracks_world_{name}.pkl') for name in ['veh_all', 'veh_static', 'ped']]
    stages = OUT / 'stages'
    stages.mkdir(exist_ok=True)
    audit, point_audit, nms_audit = [], [], []
    objects = {id(all_tracks): 'veh_all', id(static_tracks): 'veh_static', id(ped_tracks): 'ped'}
    original_delete = native.delete_tracks
    original_motion = native.motion_state_refinement
    original_merge = native.merge_disjointed_tracks
    original_rolling = native.get_track_rolling_kde_interpolation
    original_propagate = native.propagate_static_boxes
    original_nms, original_points = native.nms, native.get_points_in_box

    def delete_traced(tracks, min_score, num_boxes_abv_score=3):
        before = set(tracks)
        original_delete(tracks, min_score, num_boxes_abv_score)
        audit.append(dict(stage='track_filter', group=objects.get(id(tracks), 'unknown'), score_strictly_greater_than=min_score,
                          required_box_count=num_boxes_abv_score, tracks_before=len(before), tracks_after=len(tracks),
                          removed_track_ids=sorted(before-set(tracks))))

    def motion_traced(*args, **kwargs):
        result = original_motion(*args, **kwargs)
        for name, tracks in [('veh_all', args[0]), ('veh_static', args[1])]:
            save(stages / f'{name}_after_motion.pkl', tracks)
            audit.append(dict(stage='motion_refinement', group=name, **track_stats(tracks)))
        save(stages / 'matched_static_track_ids.pkl', result)
        return result

    def merge_traced(*args, **kwargs):
        original_merge(*args, **kwargs)
        for name, tracks in [('veh_all', args[0]), ('veh_static', args[1])]:
            save(stages / f'{name}_after_merge.pkl', tracks)
            audit.append(dict(stage='merge_disjointed_tracks', group=name, **track_stats(tracks)))

    def rolling_traced(*args, **kwargs):
        original_rolling(*args, **kwargs)
        tracks = args[1]
        save(stages / 'veh_static_after_rolling_kbf.pkl', tracks)
        audit.append(dict(stage='rolling_kbf', **track_stats(tracks)))

    def propagate_traced(*args, **kwargs):
        original_propagate(*args, **kwargs)
        tracks = args[1]
        counts = Counter()
        for value in tracks.values():
            base = value['frameid_to_box']
            start, end = min(base), max(base)
            for fid in value['frameid_to_propboxes']:
                counts['existing_frame' if fid in base else 'internal_gap' if start <= fid <= end else 'external_propagation'] += 1
        save(stages / 'veh_static_after_propagation.pkl', tracks)
        audit.append(dict(stage='propagate_boxes', **track_stats(tracks), frame_categories=dict(counts)))

    def nms_traced(boxes, scores, thresh=.05):
        mask = original_nms(boxes, scores, thresh)
        nms_audit.append(dict(threshold=thresh, candidates=len(boxes), retained=int(mask.sum())))
        return mask

    def points_traced(points, box):
        result = original_points(points, box)
        point_audit.append(dict(frame_id=dataset.current_frame, box=box.tolist(), num_points=len(result[0])))
        return result

    native.delete_tracks = delete_traced
    native.motion_state_refinement = motion_traced
    native.merge_disjointed_tracks = merge_traced
    native.get_track_rolling_kde_interpolation = rolling_traced
    native.propagate_static_boxes = propagate_traced
    try:
        all_tracks, static_tracks = native.refine_veh_labels(
            dataset, list(initial), all_tracks, static_tracks,
            static_trk_score_th=config['TRACKING']['VEH_STATIC']['RUNNING']['SCORE_TH'],
            veh_pos_th=config['PS_SCORE_TH']['POS_TH'][0], refine_cfg=config['TEMPORAL_REFINEMENT'], save_dir=str(stages))
        ped_tracks = native.refine_ped_labels(ped_tracks, config['PS_SCORE_TH']['POS_TH'][1],
                                             config['TEMPORAL_REFINEMENT']['TRACK_FILTERING'])
        save(stages / 'ped_after_refinement.pkl', ped_tracks)
        audit.append(dict(stage='pedestrian_refinement', **track_stats(ped_tracks)))
        native.nms, native.get_points_in_box = nms_traced, points_traced
        labels = native.update_ps(dataset, initial, all_tracks, static_tracks, ped_tracks,
                                  veh_pos_th=config['PS_SCORE_TH']['POS_TH'][0], veh_nms_th=.05, ped_nms_th=.5,
                                  frame2box_key_static='frameid_to_propboxes', frame2box_key='frameid_to_box', frame_ids=list(initial))
        if PRESERVE_CYCLISTS:
            # The repository's native update_ps only exports vehicles/pedestrians.
            # Keep that exact output, then explicitly preserve the third category.
            native_final = native.select_ps_by_th(copy.deepcopy(labels), config['PS_SCORE_TH']['POS_TH'])
            save(OUT / 'final_native_two_class.pkl', native_final)
            save(OUT / 'final_native_positive_only.pkl',
                 {fid:{key:array[value['gt_boxes'][:,7] > 0].copy() for key,array in value.items()}
                  for fid,value in native_final.items()})
            cyclist_audit = dict(input=0, after_nms=0, points_rejected=0, ego_rejected=0, retained=0)
            for fid, value in initial.items():
                candidates = value['gt_boxes'][abs(value['gt_boxes'][:, 7]) == 3].copy()
                cyclist_audit['input'] += len(candidates)
                if len(candidates) > 1:
                    candidates = candidates[nms_traced(candidates[:, :7], candidates[:, 8], .3)]
                cyclist_audit['after_nms'] += len(candidates)
                points = dataset.get_lidar(dataset.infos[dataset.frameid_to_idx[fid]]['lidar_path'])
                kept, counts = [], []
                for box in candidates:
                    count = len(points_traced(points, box)[0])
                    if count <= 1:
                        cyclist_audit['points_rejected'] += 1
                    elif abs(box[0]) < 1 and abs(box[1]) < 1:
                        cyclist_audit['ego_rejected'] += 1
                    else:
                        kept.append(box)
                        counts.append(count)
                cyclist_audit['retained'] += len(kept)
                labels[fid]['gt_boxes'] = np.vstack([labels[fid]['gt_boxes'], np.array(kept).reshape(-1, 9)])
                labels[fid]['num_pts'] = np.r_[labels[fid]['num_pts'], counts]
                labels[fid]['memory_counter'] = np.r_[labels[fid]['memory_counter'], np.zeros(len(kept))]
            write_json(OUT / 'cyclist_output_audit.json', cyclist_audit)
        save(stages / 'labels_before_score_selection.pkl', labels)
        before_selection = label_stats(labels)
        final = native.select_ps_by_th(labels, config['PS_SCORE_TH']['POS_TH'])
        if PRESERVE_CYCLISTS:
            # select_ps_by_th's native cyclist mask tests class2; apply class3 explicitly.
            for value in final.values():
                boxes = value['gt_boxes']
                mask = (abs(boxes[:, 7]) == 3) & (boxes[:, 8] < config['PS_SCORE_TH']['POS_TH'][2])
                boxes[mask, 7] = -3
        save(OUT / 'final_ps_dict.pkl', final)
        positive = {}
        for fid, value in final.items():
            mask = value['gt_boxes'][:, 7] > 0
            positive[fid] = {key: array[mask].copy() for key, array in value.items()}
        save(OUT / 'final_positive_only.pkl', positive)
        write_json(OUT / 'refinement_audit.json', audit)
        write_json(OUT / 'point_support_audit.json', point_audit)
        write_json(OUT / 'nms_audit.json', nms_audit)
        summary = dict(input=label_stats(initial), before_score_selection=before_selection,
                       final=label_stats(final), final_positive_only=label_stats(positive),
                       final_tracks={k:track_stats(v) for k,v in [('veh_all',all_tracks),('veh_static',static_tracks),('ped',ped_tracks)]},
                       point_checks=len(point_audit), rejected_points_le_1=sum(p['num_points'] <= 1 for p in point_audit),
                       nms_candidates=sum(n['candidates'] for n in nms_audit), nms_retained=sum(n['retained'] for n in nms_audit),
                       ego_boxes_removed=sum(p['num_points']>1 for p in point_audit)-before_selection['box_count'],
                       per_frame=[dict(frame_id=fid, input=len(initial[fid]['gt_boxes']), final=len(final[fid]['gt_boxes']),
                                       positive=len(positive[fid]['gt_boxes'])) for fid in initial],
                       native_final_nms=dict(vehicle=.05,pedestrian=.5), min_points_rule='num_pts > 1 with native XY margin 0.1m',
                       limitations=['No ground truth; output counts do not establish accuracy.',
                                    'Native frame-count thresholds unchanged at approximately20Hz.',
                                    'Source tracker/refinement behavior retained; no extra learned refinement.',
                                    'Negative class IDs in final_ps_dict mark below-threshold labels, not positive training labels.'])
        summary['nms_bypass_single_candidates'] = len(point_audit)-summary['nms_retained']
        summary['total_candidates_before_nms'] = summary['nms_candidates']+summary['nms_bypass_single_candidates']
        summary['removed_by_nms'] = summary['nms_candidates']-summary['nms_retained']
        if PRESERVE_CYCLISTS:
            summary['cyclist_output_policy'] = config['CYCLIST_OUTPUT_POLICY']
            summary['native_two_class_final'] = label_stats(native_final)
            summary['limitations'].append('Cyclist preserved from fusion with spatial/score filtering only; no native cyclist tracking/refinement.')
        write_json(OUT / 'summary.json', summary)
        print('FINAL', json.dumps(summary['final']), flush=True)
        print('POSITIVE', json.dumps(summary['final_positive_only']), flush=True)
    finally:
        native.delete_tracks = original_delete
        native.motion_state_refinement = original_motion
        native.merge_disjointed_tracks = original_merge
        native.get_track_rolling_kde_interpolation = original_rolling
        native.propagate_static_boxes = original_propagate
        native.nms, native.get_points_in_box = original_nms, original_points


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase', choices=['all', 'prepare', 'track', 'refine'], default='all')
    args = parser.parse_args()
    config = prepare()
    if args.phase in ['all', 'track']:
        track(config)
    if args.phase in ['all', 'refine']:
        refine(config)
    expected = json.loads((OUT / 'run_metadata.json').read_text())['source_sha256']
    for path, digest in expected.items():
        if fingerprint(ROOT / path) != digest:
            raise RuntimeError('Source artifact changed during the run')


if __name__ == '__main__':
    main()
