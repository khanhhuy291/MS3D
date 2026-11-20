import unittest

import numpy as np

from bevfusion_box_adapter import corners_to_vehicle_boxes
from fuse_three_models import fuse_frame, fuse_members, DEFAULT_CONFIG


def row(score, x=0, name='car'):
    return dict(name=np.array([name]), score=np.array([score]),
                boxes_lidar=np.array([[x, 0, .8, 4, 2, 1.6, .1]], dtype=np.float32))


def keep_all(boxes, scores, threshold):
    return np.ones(len(boxes), dtype=bool)


def legacy_corners():
    corners = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
                        [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]], dtype=float)
    corners = (corners-[.5, .5, 0])*[2, 4, 1.6]
    c, s = np.cos(.6), np.sin(.6)
    return (corners @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])+[1, 2, -2])[None]


class ThreeModelChecks(unittest.TestCase):
    def test_actual_legacy_contract_identity_has_center_z_and_opposite_yaw(self):
        boxes, _ = corners_to_vehicle_boxes(legacy_corners(), np.eye(4))
        np.testing.assert_allclose(boxes[0], [1, 2, -1.2, 2, 4, 1.6, -.6], atol=1e-6)

    def test_tilted_extrinsic_enclosure_contains_all_eight_corners(self):
        angle = .15
        transform = np.array([[np.cos(angle), 0, np.sin(angle), 1.69], [0, 1, 0, 0],
                              [-np.sin(angle), 0, np.cos(angle), 1.81], [0, 0, 0, 1]])
        boxes, transformed = corners_to_vehicle_boxes(legacy_corners(), transform)
        center, dims, yaw = boxes[0, :3], boxes[0, 3:6], boxes[0, 6]
        basis = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        local = (transformed[0]-center) @ basis
        self.assertTrue((abs(local) <= dims/2+1e-5).all())
        expected_center = np.array([1, 2, -1.2]) @ transform[:3, :3].T+transform[:3, 3]
        np.testing.assert_allclose(center, expected_center, atol=1e-6)

    def test_three_agreeing_sources_form_one_cluster(self):
        output, _ = fuse_frame([row(.8), row(.7), row(.9)], DEFAULT_CONFIG,
                               lambda a, b: np.ones((len(a), len(b))), keep_all)
        self.assertEqual(output['source_indices'].tolist(), [[0, 0, 0]])
        self.assertEqual(output['provenance'], ['lyft+nuscenes+bevfusion'])
        self.assertAlmostEqual(output['gt_boxes'][0, 8], (.8**2+.7**2+.9**2)/2.4, places=6)

    def test_custom_bcv_source_names_and_indices_are_not_legacy_names(self):
        config = dict(DEFAULT_CONFIG, source_order=['voxelrcnn', 'centerpoint', 'bevfusion'], align_all_pairs=True)
        output, _ = fuse_frame([row(.8), row(.7), row(.9)], config,
                               lambda a,b: np.ones((len(a),len(b))), keep_all)
        self.assertEqual(output['provenance'], ['voxelrcnn+centerpoint+bevfusion'])
        self.assertEqual(output['source_indices'].tolist(), [[0,0,0]])

    def test_bcv_seed_pair_aligns_equivalent_axes_without_third_source(self):
        a = np.array([0,0,.8,4,2,1.6,0,1,.9])
        b = np.array([0,0,.8,2,4,1.6,np.pi/2,1,.8])
        output = fuse_members([dict(source=0,box=a),dict(source=1,box=b)], align_all_pairs=True)
        np.testing.assert_allclose(output[3:6], [4,2,1.6], atol=1e-6)

    def test_bcv_cyclist_category_is_fused_without_mapping_to_vehicle(self):
        config = dict(DEFAULT_CONFIG, source_order=['voxelrcnn', 'centerpoint', 'bevfusion'], align_all_pairs=True)
        output, _ = fuse_frame([row(.8,name='bicycle'),row(.9,name='motorcycle'),row(.8,name='motorcycle')], config,
                               lambda a,b: np.ones((len(a),len(b))), keep_all)
        self.assertEqual(output['gt_boxes'][:,7].tolist(), [3])

    def test_third_source_must_agree_with_both_existing_members(self):
        def iou(a, b):
            return np.ones((1, 1))*.9 if len(a) == 1 else np.array([[.9], [.1]])
        output, _ = fuse_frame([row(.8), row(.7), row(.9)], DEFAULT_CONFIG, iou, keep_all)
        self.assertEqual(output['source_indices'].tolist(), [[0, 0, -1], [-1, -1, 0]])

    def test_bevfusion_strong_singleton_survives_distant_sources(self):
        output, _ = fuse_frame([row(.1, 0), row(.1, 20), row(.95, 40)], DEFAULT_CONFIG,
                               lambda a, b: np.zeros((len(a), len(b))), keep_all)
        self.assertEqual(output['source_indices'].tolist(), [[-1, -1, 0]])
        self.assertEqual(output['provenance'], ['bevfusion'])

    def test_bevfusion_cannot_be_used_for_two_clusters(self):
        a = row(.8)
        a['boxes_lidar'] = np.concatenate([a['boxes_lidar'], a['boxes_lidar']+[.5, 0, 0, 0, 0, 0, 0]])
        a['name'], a['score'] = np.array(['car', 'car']), np.array([.8, .9])
        empty = dict(boxes_lidar=np.empty((0, 7)), name=np.array([]), score=np.array([]))
        output, _ = fuse_frame([a, empty, row(.95)], DEFAULT_CONFIG,
                               lambda a, b: np.ones((len(a), len(b))), keep_all)
        self.assertEqual(int((output['source_indices'][:, 2] >= 0).sum()), 1)

    def test_construction_vehicle_maps_to_vehicle_supercategory(self):
        output, _ = fuse_frame([row(.8), row(.8), row(.95, name='construction_vehicle')], DEFAULT_CONFIG,
                               lambda a, b: np.ones((len(a), len(b))), keep_all)
        self.assertEqual(int(output['gt_boxes'][0, 7]), 1)

    def test_equivalent_width_length_representations_do_not_average_to_square(self):
        a = np.array([0, 0, .8, 4, 2, 1.6, 0, 1, .9])
        b = np.array([0, 0, .8, 2, 4, 1.6, np.pi/2, 1, .8])
        output = fuse_members([dict(source=0, box=a), dict(source=2, box=b)])
        np.testing.assert_allclose(output[3:6], [4, 2, 1.6], atol=1e-6)
        self.assertEqual(float(output[6]), 0)

    def test_stronger_bevfusion_can_keep_its_own_axis_representation(self):
        a = np.array([0, 0, .8, 4, 2, 1.6, 0, 1, .8])
        b = np.array([0, 0, .8, 2, 4, 1.6, np.pi/2, 1, .9])
        output = fuse_members([dict(source=0, box=a), dict(source=2, box=b)])
        np.testing.assert_allclose(output[3:6], [2, 4, 1.6], atol=1e-6)
        self.assertAlmostEqual(float(output[6]), np.pi/2, places=6)


if __name__ == '__main__':
    unittest.main()
