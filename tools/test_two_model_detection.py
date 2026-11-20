"""Input-contract and coordinate checks; no CUDA dependencies required."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from two_model_detection import ROOT, read_pcd


class DetectionInputChecks(unittest.TestCase):
    def test_timestamp_comes_from_header_and_binary_values_are_preserved(self):
        points = np.array([[1, 2, 3, 4], [-1, -2, -3, 5]], dtype='<f4')
        header = ('# timestamp 1789541343.994171\nVERSION 0.7\n'
                  'FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n'
                  'COUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frame_00000.pcd'
            path.write_bytes(header.encode('ascii') + points.tobytes())
            actual, timestamp = read_pcd(path)
            np.testing.assert_array_equal(actual, points)
            self.assertEqual(timestamp, 1789541343.994171)

    def test_unsupported_layout_is_rejected(self):
        header = ('# timestamp 1\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n'
                  'COUNT 1 1 1\nPOINTS 0\nDATA binary\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frame.pcd'
            path.write_bytes(header.encode('ascii'))
            with self.assertRaisesRegex(ValueError, 'Unsupported PCD'):
                read_pcd(path)

    def test_actual_pose_conjugation_and_cached_points(self):
        output = ROOT / 'runs/two_model_detection'
        manifest = json.loads((output / 'manifest.json').read_text())
        extrinsic = np.array(manifest['extrinsic'])
        first, later = manifest['frames'][0], manifest['frames'][10]
        raw, _ = read_pcd(ROOT / first['source'])
        cached = np.load(output / 'prepared' / f"{first['frame_id']}.npy")
        vehicle_from_lidar = extrinsic
        world_from_vehicle0 = np.array(first['pose'])
        world_from_vehicle1 = np.array(later['pose'])
        world_from_lidar0 = world_from_vehicle0 @ extrinsic
        world_from_lidar1 = world_from_vehicle1 @ extrinsic
        via_vehicle = np.linalg.inv(world_from_vehicle1) @ world_from_vehicle0
        via_lidar = vehicle_from_lidar @ np.linalg.inv(world_from_lidar1) @ world_from_lidar0
        a = cached[:1000] @ via_vehicle[:3, :3].T + via_vehicle[:3, 3]
        b = raw[:1000, :3] @ via_lidar[:3, :3].T + via_lidar[:3, 3]
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)
        self.assertTrue(np.all(np.diff([f['timestamp'] for f in manifest['frames']]) > 0))


if __name__ == '__main__':
    unittest.main()
