# CenterPoint Voxel1440 trên viewer local — 2026-11-16

Trạng thái: **HOÀN THÀNH**, đã thêm case thứ9 `centerpoint`. Nạp detection do người dùng đã chạy tại `runs/centerpoint_voxel1440_fm_singleframe`, không chạy lại inference.

## Input và cách xử lý

- Đọc `summary.json` và204file `predictions/frame_XXXXX.json` trực tiếp; không cần tạo pickle/cache trung gian.
- Xác nhận frame ID, thứ tự, timestamp và số điểm tương ứng bộ204frame prepared hiện có.
- JSON `boxes` có thứ tự `[x,y,z,length,width,height,vx,vy,yaw]`, gravity center, hệ `model_lidar`. Các box này đã được runner CenterPoint chuyển từ native width/length/yaw sang length-axis yaw; không chuyển lại `boxes_centerpoint_native`.
- Tính `T_vehicle_model = T_vehicle_FM @ inverse(T_model_FM)` từ extrinsic manifest và `input_to_model` của summary. Kết quả R=I, t=[0,0,1.8] trong sai số số học. Viewer cộng1.8m vào Z tâm, giữ nguyên XY/dimensions/yaw; không áp dụng extrinsic FM thêm lần nữa.
- Point cloud vẫn dùng prepared vehicle XYZ chung, vì các frame ID/timestamp/point count đều khớp. Không cần Docker/GPU inference để mở viewer.
- Kiểm tra contract khi load; báo lỗi nếu frame/transform/schema/class/score không tương thích.

## Lớp, score và bộ lọc

Giữ đủ47,020box nguồn đã lưu từ ngưỡng0.1. Slider mặc định0.3 cho11,156box trên toàn sequence:

| Nhóm viewer | Số box score>=0.3 | Lớp gốc |
|---|---:|---|
| Vehicle |10,041|car/truck/bus/construction_vehicle/trailer|
| Pedestrian |120|pedestrian|
| Cyclist |994|motorcycle/bicycle|
| Barrier |1|barrier|
| Traffic cone |0|traffic_cone|
| Tổng |11,156| |

Không bỏ barrier/traffic cone khỏi dữ liệu. Thêm hai bộ lọc và màu lớp cho chúng. Tên lớp gốc được giữ trong `original_name`, hiện ở inspector khi chọn box và có trong JSON export. Mỗi box có source `centerpoint`, chỉ số nguồn trỏ về hàng JSON gốc. Velocity chưa hiển thị thành tracking/trajectory; đây là single-frame detection.

## Mở viewer

Nếu đang chạy server phiên bản cũ, dừng bằng Ctrl+C rồi chạy lại:

```bash
cd /home/khanhhuy/MS3D
python3 tools/serve_local_viewer.py --port 8765
```

Mở `http://127.0.0.1:8765/?frame=100&case=centerpoint&score=0.3`.

Chọn **CenterPoint · Voxel1440 · 1 frame** ở cột trái; hai cửa sổ BEV/3D dùng cùng box. Bật Đối chiếu, chọn `fusion3`, `nuscenes` hoặc `ms3d_final` để overlay. Khi so nhiều model, đặt bộ lọc nguồn Tất cả để không vô tình ẩn box của model còn lại. Nếu8765 đang được dùng, có thể chọn `--port 8766` và đổi URL tương ứng.

## Bằng chứng kiểm tra

- **7 test data/artifact contract PASS**, gồm toàn204frame CenterPoint: bảo toàn XY/size/yaw/score/name, Z+1.8m, class mapping, totals khớp verification và summary nguồn.
- **12 nhóm QA Chrome/WebGL regression PASS** với9case, gồm CenterPoint, chuyển frame, score/class/source, overlay, camera, playback, export, responsive1024/390; không lỗi JS/console hoặc request ngoài localhost.
- **8 kiểm tra riêng CenterPoint PASS**: link trực tiếp, frame100=53box ở0.3; slider0.1 hiện đủ box của frame; source filter; overlay44box fusion3; inspector lớp gốc; barrier frame199 lọc ra1box; mobile390 không tràn ngang.
- SHA256 summary và204JSON nguồn giữ nguyên sau tích hợp/QA. Không sửa kết quả detector, fusion hoặc pseudo-label cũ.
- QA server8766 và Chrome tạm đã dừng sau kiểm tra; dùng lệnh ở trên để mở viewer.

Bằng chứng tại `runs/local_viewer/centerpoint_qa`:

- [data_verification.json](../../../runs/local_viewer/centerpoint_qa/data_verification.json).
- [browser_verification.json](../../../runs/local_viewer/centerpoint_qa/browser_verification.json).
- [centerpoint_browser_verification.json](../../../runs/local_viewer/centerpoint_qa/centerpoint_browser_verification.json).
- [Ảnh BEV/3D CenterPoint + fusion3 frame100](../../../runs/local_viewer/centerpoint_qa/centerpoint_frame100_overlay.png).

Model theo summary được train10sweep nhưng run hiện tại chỉ1sweep. Số box/visualization không phải accuracy hoặc mAP/NDS; chưa có GT để so chất lượng detector.
