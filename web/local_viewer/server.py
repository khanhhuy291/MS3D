"""Read-only local API for the recorded MS3D experiments. Requires only NumPy."""
import argparse
import json
import mimetypes
import pickle
import re
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent
SOURCES = ['lyft', 'nuscenes', 'bevfusion']
SOURCE_ORDERS = {'fusion': SOURCES, 'fusion_bcv': ['voxelrcnn', 'centerpoint', 'bevfusion']}
LABELS = {'car': 1, 'truck': 1, 'bus': 1, 'construction_vehicle': 1,
          'trailer': 1, 'pedestrian': 2, 'motorcycle': 3, 'bicycle': 3,
          'barrier': 4, 'traffic_cone': 5}
CASE_FILES = [
    ('lyft', 'Lyft · PV-RCNN++', 'runs/two_model_detection/lyft/result.pkl', 'raw',
     'XYZ · 1 sweep · prediction lưu từ score 0.1'),
    ('nuscenes', 'nuScenes · VoxelRCNN', 'runs/two_model_detection/nuscenes/result.pkl', 'raw',
     'XYZ + time · tối đa 10 sweeps · prediction lưu từ score 0.1'),
    ('bevfusion', 'BEVFusion · LiDAR-only', 'runs/three_model_detection/bevfusion/result.pkl', 'raw',
     'XYZI + time · 1 sweep · intensity gốc bằng 0 · box đã đổi sang vehicle'),
    ('centerpoint', 'CenterPoint · Voxel1440 · 1 frame',
     'runs/centerpoint_voxel1440_fm_singleframe/summary.json', 'centerpoint',
     'FM · 1 sweep · score từ 0.1 · model_lidar → vehicle: Z +1.8m · giữ nhãn lớp gốc'),
    ('kbf2', 'KBF gốc · 2 model', 'runs/two_model_detection/fusion/kbf.pkl', 'kbf',
     'Lọc gốc score 0.3 · loại box không đủ hỗ trợ · không lưu provenance chính xác'),
    ('fusion2', 'Fusion 2 · singleton ≥ 0.7', 'runs/two_model_detection_singletons/fusion.pkl', 'fusion',
     'Lyft + nuScenes · cặp ≥ 0.3 · singleton ≥ 0.7'),
    ('fusion3', 'Fusion 3 · singleton ≥ 0.7', 'runs/three_model_detection/fusion.pkl', 'fusion',
     'Lyft + nuScenes + BEVFusion · nhóm ≥ 0.3 · singleton ≥ 0.7'),
    ('ms3d_final', 'MS3D++ · pseudo-label cuối',
     'runs/ms3d_three_model_round1/final_positive_only.pkl', 'pseudo',
     'Sau tracking/refinement · Vehicle ≥ 0.7 · Pedestrian ≥ 0.6 · chỉ nhãn dương · không lưu nguồn từng box'),
    ('fusion_bcv', 'Fusion · BEV + CenterPoint + VoxelRCNN',
     'runs/fusion_bcv/fusion.pkl', 'fusion_bcv',
     'Bộ ba mới · singleton ≥ 0.7 · nhóm ≥ 0.3 · Vehicle/Pedestrian/Cyclist'),
    ('ms3d_bcv', 'MS3D++ · bộ ba BCV · nhãn cuối',
     'runs/ms3d_bcv_round1/final_positive_only.pkl', 'pseudo_bcv',
     'V/P đã tracking/refine · Cyclist chỉ lọc fusion · positive V≥0.7/P≥0.6/C≥0.5'),
    ('fusion2_relaxed', 'Fusion 2 · singleton ≥ 0.5',
     'runs/two_model_detection_singletons/sensitivity_0p5/fusion.pkl', 'fusion',
     'Cấu hình phụ · Lyft + nuScenes · cặp ≥ 0.3 · singleton ≥ 0.5'),
]


class ViewerStore:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        manifest_path = self.root / 'runs/two_model_detection/manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f'Chưa có prepared data: {manifest_path}')
        self.manifest = json.loads(manifest_path.read_text())
        self.frames = self.manifest['frames']
        self.cases, self.records, self.missing = [], {}, []
        expected_ids = [frame['frame_id'] for frame in self.frames]
        for key, title, relative, kind, description in CASE_FILES:
            path = self.root / relative
            if not path.is_file():
                self.missing.append(title)
                continue
            # Only fixed local artifact paths are unpickled; the HTTP API accepts no paths/uploads.
            if kind == 'centerpoint':
                content = self.load_centerpoint(path)
            else:
                with path.open('rb') as stream:
                    content = pickle.load(stream)
            rows = content if isinstance(content, list) else [dict(v, frame_id=k) for k, v in content.items()]
            if [row['frame_id'] for row in rows] != expected_ids:
                raise ValueError(f'Frame IDs/order không khớp: {relative}')
            self.records[key] = [self.normalize(row, key, 'raw' if kind == 'centerpoint' else kind) for row in rows]
            self.cases.append(dict(id=key, title=title, description=description,
                                   stored_min_score=.5 if kind == 'pseudo_bcv' else (.6 if kind == 'pseudo' else (.1 if kind in ['raw', 'centerpoint'] else .3)),
                                   has_provenance=kind in ['raw', 'fusion', 'fusion_bcv', 'centerpoint']))
        if not self.cases:
            raise ValueError('Không có file prediction tương thích')

    def load_centerpoint(self, path):
        """Read the supplied JSON run; transform its virtual LiDAR frame once."""
        summary = json.loads(path.read_text())
        if summary['arguments']['input_frame'] != 'fm':
            raise ValueError('CenterPoint run must use the verified FM input frame')
        vehicle_from_model = np.asarray(self.manifest['extrinsic']) @ np.linalg.inv(np.asarray(summary['input_to_model']))
        expected = np.eye(4)
        expected[2, 3] = summary['arguments']['model_sensor_height']
        if not np.allclose(vehicle_from_model, expected, atol=1e-8):
            raise ValueError('Unexpected CenterPoint model-to-vehicle transform')
        frames = self.frames
        files = sorted((path.parent / 'predictions').glob('*.json'))
        if [p.stem for p in files] != [f['frame_id'] for f in frames] or summary['frames'] != len(frames):
            raise ValueError('CenterPoint prediction frame coverage mismatch')
        rows = []
        for file, frame in zip(files, frames):
            row = json.loads(file.read_text())
            if (row['frame_id'] != frame['frame_id'] or abs(row['timestamp']-frame['timestamp']) > 1e-6
                    or row['coordinate_frame'] != 'model_lidar' or row['center'] != 'gravity'
                    or row['box_fields'] != ['x','y','z','length','width','height','vx','vy','yaw']):
                raise ValueError(f'CenterPoint frame/box contract mismatch: {file.name}')
            boxes = np.asarray(row['boxes'], dtype=float).reshape(-1, 9)
            scores, labels = np.asarray(row['scores']), np.asarray(row['labels'])
            names = row['names']
            if (len(boxes) != len(scores) or len(boxes) != len(labels) or len(boxes) != len(names)
                    or not np.isfinite(scores).all() or not np.all((scores >= 0) & (scores <= 1))
                    or any(summary['classes'][int(label)] != name for label, name in zip(labels, names))):
                raise ValueError(f'CenterPoint class/score contract mismatch: {file.name}')
            # JSON `boxes` already has length-axis yaw; do not convert native yaw again.
            output = boxes[:, [0, 1, 2, 3, 4, 5, 8]].copy()
            output[:, :3] += vehicle_from_model[:3, 3]
            rows.append(dict(frame_id=row['frame_id'], boxes_lidar=output, name=names, score=scores))
        return rows

    @staticmethod
    def normalize(row, key, kind):
        if kind == 'raw':
            boxes = np.column_stack([row['boxes_lidar'][:, :7],
                                     [LABELS[n] for n in row['name']], row['score']])
        else:
            boxes = np.asarray(row['gt_boxes'])
        if boxes.ndim != 2 or boxes.shape[1] != 9 or not np.isfinite(boxes).all():
            raise ValueError(f'Invalid boxes: {key}/{row["frame_id"]}')
        result = []
        for index, box in enumerate(boxes):
            if int(box[7]) not in [1, 2, 3, 4, 5] or not (box[3:6] > 0).all():
                raise ValueError(f'Invalid class/dimensions: {key}')
            references = {}
            if kind == 'raw':
                references[key] = index
            elif kind in SOURCE_ORDERS:
                refs = row['source_indices'][index]
                references = {SOURCE_ORDERS[kind][k]: int(j) for k, j in enumerate(refs) if j >= 0}
            members = len(references)
            group_kind = 'unknown' if kind in ['kbf', 'pseudo', 'pseudo_bcv'] else ['unknown', 'single', 'pair', 'triple'][members]
            result.append(dict(id=f'{key}:{index}', index=index, box=box[:7].astype(float).tolist(),
                               class_id=int(box[7]), score=float(box[8]), kind=group_kind,
                               sources=list(references), source_indices=references,
                               original_name=str(row['name'][index]) if kind == 'raw' else None))
        return result

    def index(self, value):
        index = int(value)
        if not 0 <= index < len(self.frames):
            raise ValueError(f'Frame phải nằm trong 0–{len(self.frames)-1}')
        return index

    def catalog(self):
        start = self.frames[0]['timestamp']
        return dict(title='MS3D · Local Inspector', coordinate_frame='vehicle', units='metres',
                    point_schema='XYZ float32 little-endian, vehicle frame; no additional transform',
                    classes=[dict(id=1, name='Vehicle', color='#41cbe8'),
                             dict(id=2, name='Pedestrian', color='#ffc76a'),
                             dict(id=3, name='Cyclist', color='#f18ebc'),
                             dict(id=4, name='Barrier', color='#a5b4fc'),
                             dict(id=5, name='Traffic cone', color='#f97316')],
                    frames=[dict(index=i, frame_id=f['frame_id'], timestamp=f['timestamp'],
                                 elapsed_seconds=f['timestamp']-start, num_points=f['num_points'])
                            for i, f in enumerate(self.frames)],
                    cases=self.cases, missing_cases=self.missing,
                    default_case='fusion3' if 'fusion3' in self.records else self.cases[0]['id'],
                    sensor_origin_vehicle=np.array(self.manifest['extrinsic'])[:3, 3].tolist())

    def frame(self, index):
        index = self.index(index)
        f = self.frames[index]
        return dict(index=index, frame_id=f['frame_id'], timestamp=f['timestamp'],
                    elapsed_seconds=f['timestamp']-self.frames[0]['timestamp'],
                    num_points=f['num_points'], coordinate_frame='vehicle',
                    cases={key: rows[index] for key, rows in self.records.items()})

    @lru_cache(maxsize=12)
    def points(self, index, max_points=60000):
        index = self.index(index)
        if not 1000 <= max_points <= 300000:
            raise ValueError('max_points phải nằm trong 1000–300000')
        path = self.root / 'runs/two_model_detection/prepared' / f"{self.frames[index]['frame_id']}.npy"
        points = np.load(path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('Prepared point cloud phải có shape (N,3)')
        points = points[np.isfinite(points).all(axis=1)]
        original_count = len(points)
        if original_count > max_points:
            points = points[np.linspace(0, original_count-1, max_points).astype(np.int64)]
        return np.ascontiguousarray(points, dtype='<f4').tobytes(), len(points), original_count


def handler_for(store):
    class ViewerHandler(BaseHTTPRequestHandler):
        server_version = 'MS3DLocalViewer/1.0'

        def reply(self, data, content_type='application/json; charset=utf-8', status=200, headers=None):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'private, max-age=3600' if content_type == 'application/octet-stream' else 'no-cache')
            self.send_header('X-Content-Type-Options', 'nosniff')
            for key, value in (headers or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            parts = urlsplit(self.path)
            path = unquote(parts.path)
            try:
                if path == '/api/catalog':
                    return self.reply(store.catalog())
                match = re.fullmatch(r'/api/frame/(\d+)', path)
                if match:
                    return self.reply(store.frame(int(match[1])))
                match = re.fullmatch(r'/api/points/(\d+)', path)
                if match:
                    query = parse_qs(parts.query)
                    maximum = int(query.get('max_points', ['60000'])[0])
                    data, count, original = store.points(int(match[1]), maximum)
                    return self.reply(data, 'application/octet-stream', headers={
                        'X-Point-Count': count, 'X-Original-Point-Count': original,
                        'X-Coordinate-Frame': 'vehicle'})
                if path.startswith('/api/'):
                    return self.reply(dict(error='API không tồn tại'), status=404)
                relative = 'index.html' if path == '/' else path.lstrip('/')
                file = (STATIC / relative).resolve()
                allowed = file.is_relative_to(STATIC) and file.is_file() and file.suffix in {'.html', '.css', '.js'}
                if not allowed:
                    return self.reply(dict(error='Không tìm thấy'), status=404)
                mime = mimetypes.guess_type(str(file))[0] or 'application/octet-stream'
                return self.reply(file.read_bytes(), mime+'; charset=utf-8')
            except (ValueError, IndexError) as error:
                return self.reply(dict(error=str(error)), status=400)
            except Exception:
                import traceback
                traceback.print_exc()
                return self.reply(dict(error='Lỗi đọc artifact; xem log server'), status=500)

    return ViewerHandler


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    store = ViewerStore()
    server = ThreadingHTTPServer((args.host, args.port), handler_for(store))
    print(f'MS3D Local Inspector: http://{args.host}:{server.server_port}', flush=True)
    print(f'{len(store.frames)} frames · {len(store.cases)} results · vehicle frame · read-only', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped local viewer.', flush=True)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
