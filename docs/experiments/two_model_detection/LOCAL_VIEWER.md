# Web local để xem BEV và 3D

Trạng thái **HOÀN THÀNH 2026-11-05**: 5 test dữ liệu PASS, 12 nhóm kiểm tra Chrome/WebGL PASS, có screenshot và kiểm tra export JSON/PNG thực tế. Server QA tạm được dừng sau kiểm tra; dùng lệnh dưới đây để mở viewer khi cần.

## Chạy

```bash
cd /home/khanhhuy/MS3D
python3 tools/serve_local_viewer.py --port 8765
```

Mở **http://127.0.0.1:8765** bằng Chrome/Chromium hoặc trình duyệt hỗ trợ WebGL. Giữ terminal chạy server; `Ctrl+C` để dừng. Có thể dùng `--port 8766` nếu cổng mặc định đang được dùng. Khi dừng server, trang không tải được frame mới.

Link thẳng một frame/kết quả:

```text
http://127.0.0.1:8765/?frame=100&case=fusion3&score=0.3
```

Chỉ cần Python3.9+ có NumPy (môi trường host hiện tại đã có). Không cần Docker/GPU để mở viewer; WebGL dùng GPU trình duyệt hoặc software renderer tùy máy. Server mặc định bind `127.0.0.1`, chạy trên máy local. Các file Three.js/OrbitControls đã lưu kèm trong repo, không có request CDN/font/data bên ngoài khi sử dụng.

## Các kết quả được nạp

| ID | Trường hợp | Artifact |
|---|---|---|
| `lyft` | PV-RCNN++ Lyft | `runs/two_model_detection/lyft/result.pkl` |
| `nuscenes` | VoxelRCNN nuScenes | `runs/two_model_detection/nuscenes/result.pkl` |
| `bevfusion` | BEVFusion LiDAR-only | `runs/three_model_detection/bevfusion/result.pkl` |
| `kbf2` | KBF gốc hai model | `runs/two_model_detection/fusion/kbf.pkl` |
| `fusion2` | Fusion hai model, singleton >=0.7 | `runs/two_model_detection_singletons/fusion.pkl` |
| `fusion3` | Fusion ba model, singleton >=0.7 | `runs/three_model_detection/fusion.pkl` |
| `fusion2_relaxed` | Cấu hình phụ hai model, singleton >=0.5 | `runs/two_model_detection_singletons/sensitivity_0p5/fusion.pkl` |

204 frame và point cloud lấy từ manifest/prepared của lần thử trước. Tất cả được kiểm tra cùng frame ID/thứ tự trước khi server phục vụ. Case không có artifact sẽ được bỏ khỏi catalog; artifact lệch frame sẽ báo lỗi thay vì ghép sai frame.

## Thao tác

- Chọn kết quả ở sidebar; BEV và 3D cập nhật đồng thời.
- Timeline, số frame và nút trước/sau để duyệt; `←`/`→` khi không nhập liệu, `Space` phát/dừng. Tốc độ là mức yêu cầu; thực tế còn phụ thuộc tải/render frame.
- BEV: cuộn chuột zoom quanh con trỏ, kéo pan, bấm box để chọn.
- 3D: kéo trái xoay, kéo phải pan, cuộn zoom. Preset Phối cảnh/Trên/Trước/Trái; `R` đặt lại camera. Double-click box hoặc nút trong inspector đưa box vào giữa góc nhìn 3D.
- Lọc score, Vehicle/Pedestrian/Cyclist, singleton/nhóm2/nhóm3 và nguồn đóng góp.
- Bật **Đối chiếu** để vẽ thêm kết quả thứ hai bằng nét tím trên cả hai góc nhìn. Đây là overlay cùng frame, không trộn lại prediction.
- Bấm box trên canvas hoặc một hàng trong bảng để xem center, DX/DY/DZ, score, yaw và source index. Nguồn đóng góp của KBF gốc không được lưu nên không được giả định lại.
- Điểm hiển thị tối đa20k/60k/120k, lọc Z, kích thước điểm, bật/tắt point cloud/lưới/score trên BEV.
- Xuất **JSON** chứa box đang qua bộ lọc, frame/timestamp, hệ tọa độ, filter và overlay nếu bật. Xuất **Ảnh BEV+3D** ghép hai canvas thành PNG. File được tải về theo cơ chế Downloads của trình duyệt.

## Quy ước visualization

- **Tất cả ở hệ vehicle, đơn vị mét**: X trước, Y trái, Z lên. Viewer không áp dụng thêm extrinsic hoặc Z shift.
- Point cloud là **một sweep hiện tại** từ `prepared/frame_XXXXX.npy`, không phải cloud tích lũy10sweeps của VoxelRCNN. Lấy mẫu đều theo chỉ số nếu vượt ngân sách điểm; lọc Z chỉ áp dụng point cloud, không thay đổi box.
- BEV đặt X hướng lên, Y sang trái; 3D có axes X đỏ/Y xanh lá/Z xanh dương, ground grid tại Z0.
- Box `[x,y,z,dx,dy,dz,yaw]` dùng Z tâm. Marker trục DX và yaw là hướng của trục local-X biểu diễn box; với BEVFusion legacy nó có thể là trục rộng, không được tự suy ra hướng đầu xe.
- Màu lớp: cyan Vehicle, vàng Pedestrian, hồng Cyclist. Nét đứt là một nguồn/singleton; nét liền là nhóm nhiều nguồn hoặc KBF không lưu provenance. Overlay dùng tím.
- Thanh score là **lọc output đã lưu**. Fusion đã loại cặp dưới0.3/singleton dưới0.7 (hoặc0.5 ở cấu hình phụ), nên giảm slider không phục hồi các box đã loại. UI có thông báo khi slider thấp hơn floor của artifact.
- Bộ lọc nguồn dùng provenance chính xác khi artifact có source_indices. KBF gốc có kind `unknown`; chọn tất cả hoặc “Nguồn chưa được lưu” để xem các box đó.
- Bộ đếm là prediction qua bộ lọc, không phải accuracy. BEVFusion nhận intensity0 trong dữ liệu này; viewer có nhắc giới hạn khi chọn case đó.

## Code và API

- [Lệnh khởi động](../../../tools/serve_local_viewer.py)
- [Backend](../../../web/local_viewer/server.py)
- [HTML](../../../web/local_viewer/index.html), [CSS](../../../web/local_viewer/style.css), [JS](../../../web/local_viewer/app.js)
- `GET /api/catalog`: frame/case/class metadata.
- `GET /api/frame/100`: toàn bộ kết quả của cùng frame, gồm geometry, score, kind, sources và source_indices.
- `GET /api/points/100?max_points=60000`: XYZ float32 little-endian, header có số điểm và frame tọa độ. Giới hạn query1000–300000.

Backend chỉ đọc các path artifact cố định. HTTP không nhận upload hoặc path pickle tùy ý; static server chỉ phục vụ HTML/CSS/JS bên trong thư mục viewer. Các file kết quả detector/fusion vẫn giữ nguyên. Xuất file diễn ra trong browser, không sửa artifacts.

Server nạp predictions lúc khởi động; khởi động lại server nếu chạy lại experiments và muốn đọc artifact mới.

## Kiểm tra

```bash
python3 tools/test_local_viewer.py -v
```

5 test đã PASS: catalog204frame/7case, số box normalized khớp báo cáo đã xác minh, binary points giữ nguyên hệ vehicle, từ chối index/budget sai, provenance singleton/pair/triple/unknown.

QA browser dùng Chrome headless thật qua CDP và WebGL SwiftShader. Dependency `websocket-client` chỉ phục vụ QA, không cần khi mở viewer. Code QA: [test_local_viewer_browser.py](../../../tools/test_local_viewer_browser.py). Log/screenshots sẽ được lưu trong `runs/local_viewer/qa`.

12 nhóm kiểm tra browser đã PASS:

1. HTTP catalog/frame/binary points, dữ liệu không hợp lệ và giới hạn static path.
2. Chrome tải module local, tạo WebGL scene và BEV; frame100/fusion3 có đúng44 box theo API.
3. Lọc score, lớp, singleton và nguồn, đối chiếu chính xác số box với dữ liệu API.
4. Đổi case và overlay kết quả thứ hai trên cùng frame.
5. Chọn box từ bảng và bằng pointer thực trên BEV, inspector, focus/preset camera.
6. Xoay OrbitControls bằng pointer thực.
7. Đổi frame liên tiếp, hủy request cũ, nút trước/sau.
8. Đổi ngân sách điểm và lọc Z cập nhật cả hai góc nhìn.
9. Playback đổi frame và dừng.
10. Bấm export JSON/PNG và kiểm tra file download thực, JSON có đúng44 detection/hệ vehicle.
11. Viewport1024 và390 không tràn ngang toàn trang, WebGL vẫn hoạt động.
12. Không có Runtime.exceptionThrown/console error; không có request HTTP(S) ngoài localhost của viewer.

QA đã sửa camera Top giữ Z-up nhất quán với OrbitControls, chỉnh selector CSS để ô FPS không chiếm chiều rộng timeline, và tách thư mục download theo lần test. Không thay đổi artifact detector/fusion.

Artifacts kiểm tra:

- [Ảnh giao diện desktop](../../../runs/local_viewer/qa/desktop.png)
- [Viewport1024](../../../runs/local_viewer/qa/viewport_1024.png), [viewport390](../../../runs/local_viewer/qa/viewport_390.png)
- [Báo cáo browser verification](../../../runs/local_viewer/qa/browser_verification.json)
- [Log lần kiểm tra cuối](../../../runs/local_viewer/qa/browser_test_final_retry1.log)
- File export thực nằm dưới `runs/local_viewer/qa/downloads/`; đường dẫn chính xác được ghi trong browser_verification.json.

Phép kiểm tra browser xác nhận khả năng render/thao tác, không đánh giá độ chính xác của detection. Số44 box và59,352 điểm hiển thị trong snapshot QA là dữ liệu frame100 với mặc định score0.3/ngân sách60k/Z[-3,6], không phải hằng số dùng trong UI.


## Cập nhật 2026-11-06: nhãn cuối MS3D++

Thêm case thứ8 `ms3d_final` từ `runs/ms3d_three_model_round1/final_positive_only.pkl`: 7,205 nhãn dương/204frame. Ngưỡng đã áp là Vehicle>=0.7, Pedestrian>=0.6. Không hiển thị1,852 ignore và không suy diễn provenance từng detector sau temporal refinement. Chọn bộ lọc nguồn Tất cả; overlay `fusion3` để so trước/sau.

Nếu đang chạy server cũ, khởi động lại để nạp catalog mới, hoặc dùng `--port 8766`. Link: `http://127.0.0.1:8766/?frame=123&case=ms3d_final&score=0.3`.

6 test data-contract PASS. Report chi tiết: [FINAL_PSEUDO_LABELS.md](FINAL_PSEUDO_LABELS.md).

12 nhóm QA Chrome/WebGL PASS với8case; kết quả lần này ở `runs/ms3d_three_model_round1/viewer_qa`, không ghi đè QA trước. Ảnh đối chiếu final/fusion đã lưu cho frame115/123/197/201. Server QA8766 đã được dừng sau kiểm tra.


## Cập nhật 2026-11-16: CenterPoint

Thêm case thứ9 `centerpoint` từ `runs/centerpoint_voxel1440_fm_singleframe/predictions`: 204frame,47,020box đã lưu;11,156box khi score>=0.3. JSON gravity-center/model_lidar được đổi sang vehicle bằng Z+1.8m đã xác minh từ extrinsic và summary. Giữ original class trong inspector/export; thêm Barrier/Traffic cone và bộ lọc nguồn CenterPoint.

Khởi động lại server để nạp case mới. Link thẳng `http://127.0.0.1:8765/?frame=100&case=centerpoint&score=0.3`. Frame100=53box CenterPoint; có thể overlay44box fusion3.

7 data-contract tests,12 nhóm Chrome regression và8 kiểm tra riêng CenterPoint PASS. QA riêng ở `runs/local_viewer/centerpoint_qa`. Chi tiết: [CENTERPOINT_VIEWER.md](CENTERPOINT_VIEWER.md).


## Cập nhật 2026-11-16: fusion/refinement bộ ba BCV

Có11case: thêm `fusion_bcv` (BEVFusion+CenterPoint+VoxelRCNN,9,522box) và `ms3d_bcv` (6,245positive cuối). Vehicle/Pedestrian đã native tracking/refine; Cyclist chỉ NMS/point/score filtering từ fusion, không temporal refinement. Final không lưu source membership từng box.

Khởi động lại viewer, mở `http://127.0.0.1:8765/?frame=100&case=ms3d_bcv&score=0.3`. Overlay `fusion_bcv` để so trước/sau hoặc `ms3d_final` để so bộ ba cũ. Filter nguồn Tất cả; filter `VoxelRCNN · fusion BCV` áp dụng provenance nguồn mới.

8test data-contract và12nhóm Chrome regression PASS. Bằng chứng/ảnh so sánh tại `runs/local_viewer/bcv_qa`; report pipeline: [BCV_MS3D.md](BCV_MS3D.md).
