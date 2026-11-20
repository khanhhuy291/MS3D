import unittest

import numpy as np

from fuse_two_models_with_singletons import DEFAULT_CONFIG, fuse_frame, fuse_pair, select_pairs, select_unmatched


def row(boxes, scores):
    return dict(name=np.array(['car']*len(scores)),
                boxes_lidar=np.array(boxes, dtype=np.float32).reshape(-1, 7),
                score=np.array(scores))


class PairFusionChecks(unittest.TestCase):
    def test_assignment_preserves_two_valid_pairs_in_ambiguous_case(self):
        iou = np.array([[.9, .8], [.85, .1]])
        pairs = select_pairs(iou, np.zeros((2, 2)), np.zeros((2, 2)), .3, 2, 1)
        self.assertEqual(set(map(tuple, pairs)), {(0, 1), (1, 0)})

    def test_spatial_gates_prevent_merging_adjacent_or_separate_height_objects(self):
        iou = np.array([[.0, .8, .8, .8]])
        pairs = select_pairs(iou, np.array([[.5, 2.1, .5, .5]]),
                             np.array([[0, 0, 1.1, 0]]), .3, 2, 1)
        np.testing.assert_array_equal(pairs, [[0, 3]])

    def test_empty_assignment(self):
        self.assertEqual(select_pairs(np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)), .3, 2, 1).shape, (0, 2))

    def test_two_box_rule_keeps_strong_center_and_heading_across_pi(self):
        a = np.array([1, 2, .8, 4, 2, 1.6, 3.13, 1, .9])
        b = np.array([1.4, 2, .8, 5, 2, 1.6, -3.13, 1, .1])
        result = fuse_pair(a, b)
        np.testing.assert_allclose(result[:3], a[:3])
        self.assertAlmostEqual(result[6], 3.13, places=5)
        self.assertAlmostEqual(result[3], 4.1, places=5)
        self.assertAlmostEqual(result[8], .82, places=5)

    def test_matched_indices_are_not_retained_again_as_singletons(self):
        boxes = np.zeros((3, 9))
        boxes[:, 8] = [.96, .8, .69]
        self.assertEqual(select_unmatched(boxes, [1], .7), [0])

    def test_high_score_singleton_survives_without_any_other_model(self):
        a = row([[3.566, 6.137, .786, 4, 2, 1.6, 0]], [.96])
        result, _ = fuse_frame([a, row([], [])], DEFAULT_CONFIG,
                              lambda a, b: np.zeros((len(a), len(b))),
                              lambda boxes, scores, threshold: np.ones(len(boxes), dtype=bool))
        self.assertEqual(result['provenance'], ['lyft_singleton'])
        np.testing.assert_allclose(result['gt_boxes'][0, :7], a['boxes_lidar'][0])

    def test_low_score_partner_can_support_strong_box_without_duplicate_singleton(self):
        box = [1, 2, .8, 4, 2, 1.6, 0]
        result, counters = fuse_frame([row([box], [.8]), row([box], [.15])], DEFAULT_CONFIG,
                                      lambda a, b: np.ones((len(a), len(b))),
                                      lambda boxes, scores, threshold: np.ones(len(boxes), dtype=bool))
        self.assertEqual(result['provenance'], ['matched'])
        self.assertEqual(counters['matched_pairs'], 1)
        self.assertEqual(len(result['gt_boxes']), 1)

    def test_weak_singleton_is_not_promoted(self):
        box = [1, 2, .8, 4, 2, 1.6, 0]
        result, _ = fuse_frame([row([box], [.69]), row([], [])], DEFAULT_CONFIG,
                              lambda a, b: np.zeros((len(a), len(b))),
                              lambda boxes, scores, threshold: np.ones(len(boxes), dtype=bool))
        self.assertEqual(result['gt_boxes'].shape, (0, 9))


if __name__ == '__main__':
    unittest.main()
