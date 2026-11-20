import unittest

import numpy as np

from run_ms3d_from_three_model_fusion import ManifestDataset, canonicalize_vehicle_boxes


def corners(box):
    x, y, z, dx, dy, dz, yaw = box[:7]
    points = np.array([[dx/2,dy/2],[dx/2,-dy/2],[-dx/2,-dy/2],[-dx/2,dy/2]])
    points = points @ np.array([[np.cos(yaw), np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])+[x,y]
    return points[np.lexsort((points[:,1],points[:,0]))]


class BridgeChecks(unittest.TestCase):
    def test_vehicle_axis_normalization_keeps_footprint_center_and_score(self):
        original = np.array([[3,6,.8,2,4,1.7,.2,1,.96],[-3,2,.8,4,2,1.6,4,1,.8]],dtype=float)
        actual, changed = canonicalize_vehicle_boxes(original)
        self.assertEqual(changed.tolist(),[True,False])
        for a,b in zip(original,actual):
            np.testing.assert_allclose(corners(a),corners(b),atol=1e-8)
            np.testing.assert_array_equal(a[[0,1,2,5,7,8]],b[[0,1,2,5,7,8]])
        self.assertTrue((actual[:,3]>=actual[:,4]).all())

    def test_pedestrian_geometry_is_not_canonicalized(self):
        original = np.array([[1,2,.9,.4,.8,1.8,.2,2,.7]],dtype=float)
        actual, changed = canonicalize_vehicle_boxes(original)
        self.assertFalse(changed.any())
        np.testing.assert_allclose(actual,original,atol=1e-10)

    def test_native_bbox_format_roundtrip(self):
        from mot_3d.data_protos import BBox
        values = np.array([1,2,.8,4.5,1.9,1.6,.7,.91])
        np.testing.assert_allclose(BBox.bbox2array(BBox.array2bbox(values,input_opd_format=True),output_opd_format=True),values)

    def test_adapter_pose_timestamps_and_point_copy(self):
        from pcdet.utils import compatibility_utils as compat
        data = ManifestDataset()
        self.assertEqual(len(data.infos),204)
        dt=np.diff([i['timestamp'] for i in data.infos])
        self.assertTrue((dt>0).all())
        self.assertAlmostEqual(float(np.median(dt)),.05,places=4)
        frame='frame_00100'
        self.assertEqual(compat.get_sequence_name(data,frame),'20260916_134800_lidar')
        pose=compat.get_pose(data,frame)
        np.testing.assert_allclose(pose[3],[0,0,0,1])
        a=compat.get_lidar(data,frame)
        expected=a[0].copy();a[0]+=100
        np.testing.assert_array_equal(compat.get_lidar(data,frame)[0],expected)

    def test_native_ignore_labels_are_signed_class_ids(self):
        from pcdet.utils.ms3d_utils import select_ps_by_th
        boxes=np.array([[0,0,1,4,2,2,0,1,.69],[0,0,1,4,2,2,0,1,.71],
                        [0,0,1,.5,.5,2,0,2,.59],[0,0,1,.5,.5,2,0,2,.61]])
        result=select_ps_by_th({'frame':{'gt_boxes':boxes}},[.7,.6,.5])
        np.testing.assert_array_equal(result['frame']['gt_boxes'][:,7],[-1,1,-2,2])


if __name__=='__main__':
    unittest.main()
