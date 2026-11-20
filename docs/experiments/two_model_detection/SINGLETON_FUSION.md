# Fusion cặp hai nguồn và giữ box đơn score cao

Trạng thái: **HOÀN THÀNH ngày 2026-11-05**. Đã chạy đủ 204 frame, xuất 204 ảnh và video; 8 test và verification độc lập PASS. Đây là kết quả fusion từ predictions đã lưu, không chạy detector lại.

## Kết quả

Số box cộng qua 204 frame, không phải object duy nhất. Score đầu ra >=0.3; singleton cần >=0.7.

| Output | Vehicle | Pedestrian | Cyclist | Tổng |
|---|---:|---:|---:|---:|
| Lyft gốc | 6,634 | 346 | 1 | 6,981 |
| nuScenes gốc | 11,470 | 645 | 23 | 12,138 |
| KBF gốc | 4,619 | 314 | 2 | 4,935 |
| Fusion cặp + singleton >=0.7 | **7,274** | **357** | **0** | **7,631** |

Fusion mới tăng 2,696 box so với KBF gốc. Không diễn giải mức tăng số box thành tăng accuracy. Cyclist không còn output ở ngưỡng singleton 0.7; không được suy ra mọi lớp đều được cải thiện.

Provenance sau NMS:

- 4,865 box từ cặp Lyft–nuScenes.
- 2,598 singleton nuScenes.
- 168 singleton Lyft.

Đã tìm được 5,342 cặp ở admission 0.1; 452 cặp có fused score <0.3 bị loại. Có 7,659 ứng viên gồm cặp hợp lệ và singleton score cao; NMS loại 28, còn 7,631 box.

## Kiểm tra mức giữ lại dự đoán mạnh

Đối chiếu **các dự đoán gốc score >=0.7**, cùng supercategory và BEV IoU >=0.5 với box đầu ra (many-to-one):

| Nguồn | Dự đoán score >=0.7 | Được KBF gốc phủ | Được fusion mới phủ | Mới được phủ |
|---|---:|---:|---:|---:|
| Lyft | 2,890 | 2,635 | 2,890 | 255 |
| nuScenes | 5,045 | 2,707 | 5,045 | 2,338 |

Tổng 2,593 dự đoán score cao trước đây không được KBF gốc phủ nay được phủ. Hai model có thể dự đoán cùng một vật thể nên không cộng thành số vật thể duy nhất. Đây là **coverage so với dự đoán**, không phải recall/precision theo ground truth.

Ví dụ đã xác nhận khôi phục: `frame_00100`, nuScenes Vehicle score 0.9604567 tại `(3.5663,6.1370,0.7856) m` được giữ đúng geometry/score ban đầu, provenance `nuscenes_singleton`.

## Kiểm tra độ nhạy ngưỡng singleton

Chạy thêm cùng thuật toán với singleton >=0.5 vào `runs/two_model_detection_singletons/sensitivity_0p5`, không render ảnh/video cho cấu hình phụ:

| Ngưỡng singleton | Vehicle | Pedestrian | Cyclist | Tổng |
|---|---:|---:|---:|---:|
| 0.7 — cấu hình chính | 7,274 | 357 | 0 | 7,631 |
| 0.5 — cấu hình phụ | 9,485 | 369 | 4 | 9,858 |

Cấu hình 0.5 giữ nhiều box score vừa hơn, có thể giữ thêm false positives. Chưa chọn cấu hình này làm mặc định vì chưa có GT. Thêm ứng viên cũng có thể làm NMS chọn một box singleton thay cho một box matched.

## Phương pháp được chọn

1. Dùng lại predictions của hai detector trong `runs/two_model_detection`, không chạy detector lại.
2. Cho box score >=0.1 tham gia matching, quy đổi cùng supercategory.
3. Matching một-một **giữa hai model** bằng assignment có điều kiện IoU BEV, khoảng cách tâm XY và độ lệch Z. Không fusion hai box của cùng model với nhau.
4. Với cặp khớp: dùng quy tắc hai box đã được viết rõ, tương ứng nhánh special-case dự định trong KBF: giữ center/yaw/class của box score cao hơn; kích thước và score lấy trung bình có trọng số score. Không dùng nhánh tuple-shape bị lỗi trong source KBF.
5. Cặp fusion cần score đầu ra >=0.3. Box không được matching vẫn được giữ nếu score >=0.7.
6. NMS theo supercategory, ngưỡng Vehicle=0.5, Pedestrian=0.3, Cyclist=0.3.
7. Lưu provenance (`matched`, `lyft_singleton`, `nuscenes_singleton`) và chỉ số predictions gốc cho từng box.

## Ngưỡng matching

| Điều kiện | Vehicle | Pedestrian | Cyclist |
|---|---:|---:|---:|
| BEV IoU tối thiểu | 0.3 | 0.2 | 0.2 |
| Khoảng cách tâm XY tối đa (m) | 2.0 | 0.75 | 1.0 |
| Chênh lệch Z tối đa (m) | 1.0 | 1.0 | 1.0 |

Cả ba điều kiện phải thỏa. Assignment ưu tiên tối đa số cặp hợp lệ rồi tối đa tổng IoU. Đây là biến thể fusion hai nguồn có giữ singleton, không phải kết quả MS3D KDE grouping gốc. Các ngưỡng là lựa chọn thử nghiệm, chưa tuning bằng GT.

## Output và cách tái lập

Thư mục mới: `runs/two_model_detection_singletons`. Box giữ hệ vehicle từ thử nghiệm trước; không transform thêm. Predictions và KBF gốc được kiểm tra SHA256 trước/sau.

```bash
docker exec ms3d-fusion-20261105 python tools/test_fusion_singletons.py -v
docker exec ms3d-fusion-20261105 python tools/fuse_two_models_with_singletons.py
docker exec ms3d-fusion-20261105 python tools/verify_fusion_singletons.py

# Cấu hình phụ, không ghi đè output chính
docker exec ms3d-fusion-20261105 python tools/fuse_two_models_with_singletons.py \
  --singleton-score 0.5 --output runs/two_model_detection_singletons/sensitivity_0p5 --no-render
```

Ảnh/video bốn cột: Lyft, nuScenes, KBF gốc, fusion mới. Trong cột fusion mới, nét liền là cặp khớp và nét đứt là singleton. Màu vẫn biểu diễn supercategory như trước. Ảnh hiển thị ±40 m; tổng số box bao phủ toàn range detector.

Video được tạo bằng:

```bash
ffmpeg -hide_banner -loglevel warning -framerate 20 -start_number 0 \
  -i runs/two_model_detection_singletons/comparison/frame_%05d.png -frames:v 204 \
  -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart \
  runs/two_model_detection_singletons/comparison_20fps.mp4
```

Đã kiểm tra ffprobe: H.264, 2640×660, 204 frame, 20 FPS, 10.2 s. Nếu artifact được container tạo với owner root, chỉ đổi owner riêng thư mục kết quả mới về UID/GID tài khoản host trước khi tạo video từ host. Trong lần chạy này host là 1000:1000.

## Artifacts và xác minh

- [Video bốn cột](../../../runs/two_model_detection_singletons/comparison_20fps.mp4)
- [Ảnh frame 0](../../../runs/two_model_detection_singletons/comparison/frame_00000.png), [frame 100](../../../runs/two_model_detection_singletons/comparison/frame_00100.png), [frame 203](../../../runs/two_model_detection_singletons/comparison/frame_00203.png)
- [Summary](../../../runs/two_model_detection_singletons/summary.json), [config](../../../runs/two_model_detection_singletons/config.json), [verification](../../../runs/two_model_detection_singletons/verification.json)
- [Predictions fusion có provenance và source_indices](../../../runs/two_model_detection_singletons/fusion.pkl)
- [Runner](../../../tools/fuse_two_models_with_singletons.py), [8 test](../../../tools/test_fusion_singletons.py), [verification độc lập](../../../tools/verify_fusion_singletons.py)
- Logs trong `runs/two_model_detection/logs`: `singletons_tests.log`, `singletons_run.log`, `singletons_verification.log`, `singletons_sensitivity_0p5.log`.

Verification đã kiểm tra đủ 204 frame theo đúng thứ tự, finite/positive dimensions, source box không bị dùng nhiều lần, matching khác nguồn và đủ ba gate, công thức geometry/score cặp tính lại độc lập, identity và score singleton, decode đủ 204 PNG, đếm lại số box/provenance và SHA256 input gốc không đổi.

Đã xem trực quan frame 0/100/203: fusion mới giữ lại nhiều box đơn score cao dọc các hàng điểm xe mà KBF gốc bỏ, đặc biệt phía Y dương. Đây là nhận xét về output, chưa xác nhận các box được giữ lại đều là xe thật.

## Giới hạn

Giữ nhiều box hơn có thể khôi phục quan sát nhưng cũng giữ thêm false positives. Score hai detector chưa được calibration cho dữ liệu này. Chưa có GT để kết luận precision/recall/AP tăng.
