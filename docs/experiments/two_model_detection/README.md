# Thử detection và fusion hai model trên LiDAR FM

## Tiến độ mở rộng

Đã hoàn thành fusion ba model và MS3D++ tracking/refinement trên204frame. Nhãn cuối:7,205positive +1,852ignore; verification PASS. Xem [FINAL_PSEUDO_LABELS.md](FINAL_PSEUDO_LABELS.md), [PROGRESS.md](PROGRESS.md) và [viewer BEV/3D](LOCAL_VIEWER.md). Phần dưới ghi lại baseline hai model ban đầu.

## Phạm vi baseline ban đầu

Chạy PV-RCNN++ Lyft và VoxelRCNN CenterHead nuScenes, fusion bằng KBF và xuất ảnh BEV so sánh. Không chạy tracking, temporal refinement hoặc training.

## Dữ liệu và quy ước đã xác nhận

- Người dùng xác nhận PCD ở hệ LiDAR-top, chính là LiDAR FM trong `data_raw/calib_new2.json`.
- `trajectory_optimized.tum` là pose LiDAR, hiểu là `T_world_from_lidar` theo chuẩn TUM.
- 204 PCD binary XYZI; timestamp nằm trong header; khoảng 20 Hz, dài 10.140826 giây.
- Trajectory 497 pose, khoảng 5 Hz, bao phủ toàn bộ PCD.
- Dữ liệu gốc được giữ nguyên; runner chuyển tọa độ khi đọc.

## Phép biến đổi

Gọi `E = T_vehicle_lidar.FM`, tức `T_vehicle_from_lidar`.

```text
p_vehicle = E @ p_lidar
T_world_from_vehicle(t) = T_world_from_lidar(t) @ inverse(E)
p_vehicle_current = inverse(T_world_from_vehicle(current))
                    @ T_world_from_vehicle(past) @ p_vehicle_past
```

Pose tại timestamp PCD: nội suy tuyến tính translation và SLERP quaternion. Không ngoại suy ngoài trajectory. Không cộng thêm `SHIFT_COOR` sau extrinsic FM, vì extrinsic đã chứa translation Z = 1.809581687 m.

## Model

| Model | Checkpoint | Input |
|---|---|---|
| Lyft PV-RCNN++ ResNet AnchorHead | `lyft_uda_pv_rcnn_plusplus_resnet_anchorhead_1xyz_allcls.pth` | XYZ, 1 sweep |
| nuScenes VoxelRCNN CenterHead | `nuscenes_uda_voxel_rcnn_centerhead_10xyzt_allcls.pth` | XYZ + time lag |

Số input channel đã kiểm tra trực tiếp từ checkpoint: 3 và 4. Config mẫu phải được điều chỉnh tương ứng. Load thiếu trọng số phải dừng thay vì tiếp tục với trọng số ngẫu nhiên.

## Kế hoạch và bằng chứng

1. Audit dữ liệu, calibration, timestamp và checkpoint.
2. Dựng container riêng tương thích RTX 5060; kiểm tra CUDA operators.
3. Chuẩn bị runner, config và kiểm tra hình học/pose.
4. Smoke test một vài frame, so sánh input/output và load checkpoint.
5. Chạy sequence, fusion và xuất ảnh, thống kê, report.

**Trạng thái: hoàn thành 204/204 frame, fusion, 204 ảnh và video.** Xem [PROGRESS.md](PROGRESS.md) để biết lịch sử, [REPORT.md](REPORT.md) để xem kết quả đã đo và [COMMANDS.md](COMMANDS.md) để tái lập. Không có ground truth thì chỉ kết luận thay đổi/đánh giá trực quan, không khẳng định tăng accuracy.

Thử nghiệm tiếp theo để hạn chế mất box: [SINGLETON_FUSION.md](SINGLETON_FUSION.md), matching khác nguồn và giữ singleton score cao, đã hoàn thành với video bốn cột riêng.

Mở rộng ba model với BEVFusion LiDAR-only: [THREE_MODEL_DETECTION.md](THREE_MODEL_DETECTION.md), hoàn thành204 frame với video sáu ô và báo cáo riêng.

Web local tương tác để xem mọi trường hợp ở BEV và 3D: [LOCAL_VIEWER.md](LOCAL_VIEWER.md). Khởi động bằng `python3 tools/serve_local_viewer.py --port 8765`, mở `http://127.0.0.1:8765`.


Bộ ba mới BEVFusion + CenterPoint + VoxelRCNN đã fusion/tracking/refinement trên204frame, có6,245positive cuối (Cyclist spatial-only). Xem [BCV_MS3D.md](BCV_MS3D.md), viewer case `fusion_bcv` và `ms3d_bcv`.
