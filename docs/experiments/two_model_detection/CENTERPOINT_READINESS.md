# Khả năng chạy CenterPoint trong MS3D — 2026-11-16

Đã kiểm tra checkout hiện tại. Repo có implementation và đăng ký detector CenterPoint; chưa chạy inference CenterPoint trong lượt này.

## Source và config hiện có

- `pcdet/models/detectors/centerpoint.py`: `CenterPoint`, build module list; inference lấy `final_box_dicts` từ dense head, không có ROI refinement trong config mẫu.
- `pcdet/models/detectors/__init__.py`: đăng ký `CenterPoint` để `build_network` khởi tạo.
- `pcdet/models/dense_heads/center_head.py`: CenterHead dùng heatmap/center_z/dim/rot để dự đoán box.
- `tools/cfgs/kitti_models/uda_centerpoint.yaml`: MODEL.NAME=CenterPoint, MeanVFE + VoxelResBackBone8x + HeightCompression + BaseBEVBackbone + CenterHead. Chỉ lớp Car, XYZ, voxel0.1/0.1/0.15m; có FOV_POINTS_ONLY và SHIFT_COOR=[0,0,1.6] của setup KITTI. Không dùng nguyên các giả định đó cho FM đã chuyển sang vehicle.

## Phân biệt với model đã chạy

`tools/cfgs/nuscenes_models/uda_voxel_rcnn_centerhead.yaml` có MODEL.NAME=VoxelRCNN và ROI_HEAD=VoxelRCNNHead. Nó dùng CenterHead ở bước proposal nhưng vẫn là model hai giai đoạn. Checkpoint `nuscenes_uda_voxel_rcnn_centerhead_10xyzt_allcls.pth` đã dùng trong fusion không tự trở thành checkpoint CenterPoint độc lập khi đổi MODEL.NAME.

Trong `checkpoints/` hiện chỉ có ba file: Lyft PVRCNN++, nuScenes VoxelRCNN CenterHead và BEVFusion LiDAR-only. Chưa có cặp checkpoint + config pretrained CenterPoint độc lập được xác minh tương thích. Chưa kết luận checkpoint từ repo CenterPoint/MMDetection3D có thể load trực tiếp vào OpenPCDet/MS3D.

## Điều kiện chạy thử thật

1. Cần checkpoint pretrained CenterPoint và config gốc tương ứng; kiểm tra kiến trúc/tensor/class order, input features và sweeps. Thiếu config/feature contract thì chưa chạy để tránh load thiếu trọng số hoặc input sai.
2. Tái sử dụng PCD FM, calibration và trajectory đã xác minh. Chọn source/target frame theo config pretrained; nếu dùng vehicle prepared thì không lặp extrinsic/Z shift. Config nguồn dùng LiDAR sensor frame phải được xử lý tương ứng, không tự coi vehicle và LiDAR là một hệ.
3. Dựng config/runner riêng, kiểm tra CUDA/spconv của môi trường hiện tại, smoke test1–3frame và xác minh đầy đủ trọng số load.
4. Nếu smoke pass, chạy204frame, xuất box về vehicle với geometric center để so cùng frame với các model trước. Lưu riêng `runs/centerpoint_detection` và thêm viewer case khi có kết quả hợp lệ.
5. Giữ nguyên detection/fusion/pseudo-label baseline. Chưa có GT thì so trực quan và continuity, không kết luận accuracy từ số box.

Không chạy detector hoặc thay đổi kết quả cũ trong lượt kiểm tra khả năng này. Phần còn thiếu để thực thi là checkpoint/config CenterPoint tương thích được chọn cụ thể.


## Cập nhật sau kiểm tra ban đầu — 2026-11-16

Người dùng đã chạy CenterPoint ở repo riêng và cung cấp run `runs/centerpoint_voxel1440_fm_singleframe`. Đã xác minh schema/transform và tích hợp204frame vào viewer MS3D. Không chạy lại inference trong MS3D. Xem [CENTERPOINT_VIEWER.md](CENTERPOINT_VIEWER.md) để mở và so kết quả.
