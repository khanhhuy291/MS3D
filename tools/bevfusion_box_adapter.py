"""Geometry bridge based on actual MMDet3D corners, not a guessed yaw contract."""
import numpy as np


def corners_to_vehicle_boxes(corners_lidar, vehicle_from_lidar):
    """Fit upright OpenPCDet boxes enclosing the fully transformed eight corners.

    Corner ordering is MMDet3D LiDARInstance3DBoxes.corners. The 0->4 edge
    defines local +X. Roll/pitch cannot be represented in yaw-only boxes;
    dimensions are the enclosing extents in the projected local-X orientation.
    """
    corners = np.asarray(corners_lidar, dtype=np.float64)
    transform = np.asarray(vehicle_from_lidar, dtype=np.float64)
    if corners.shape[1:] != (8, 3) or transform.shape != (4, 4):
        raise ValueError('Expected (N,8,3) corners and (4,4) transform')
    vehicle = corners @ transform[:3, :3].T + transform[:3, 3]
    if not len(corners):
        return np.empty((0, 7), dtype=np.float32), vehicle
    edge = vehicle[:, 4]-vehicle[:, 0]
    yaw = np.arctan2(edge[:, 1], edge[:, 0])
    boxes = []
    for points, heading in zip(vehicle, yaw):
        basis = np.array([[np.cos(heading), -np.sin(heading), 0],
                          [np.sin(heading), np.cos(heading), 0], [0, 0, 1]])
        local = points @ basis
        lo, hi = local.min(axis=0), local.max(axis=0)
        center = ((hi+lo)/2) @ basis.T
        dims = hi-lo
        boxes.append(np.r_[center, dims, heading])
        if not np.isfinite(points).all() or not (dims > 0).all():
            raise ValueError('Invalid transformed box')
    return np.array(boxes, dtype=np.float32), vehicle
