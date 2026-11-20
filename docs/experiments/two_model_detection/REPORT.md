# Báo cáo thử nghiệm hai detector

Trạng thái: **HOÀN THÀNH ngày 2026-11-05**. Hai model, KBF và 204 ảnh so sánh đã chạy xong. Video và kiểm tra artifact đã pass. Đây là thử nghiệm detection/fusion, chưa có ground truth để đo accuracy.

**Cập nhật chẩn đoán mất box:** xem [KBF_DIAGNOSIS.md](KBF_DIAGNOSIS.md). Audit xác định 8,622 box đầu vào không vào nhóm hợp lệ; NMS chỉ loại 59/4,994 ứng viên. Hạ admission từ 0.3 xuống 0.1, vẫn đếm đầu ra score>=0.3, tăng tổng từ 4,935 lên 6,656. Đây là ablation phân tích riêng, không thay kết quả chính hoặc khẳng định tăng accuracy.

**Thử nghiệm tiếp theo đã hoàn thành:** [SINGLETON_FUSION.md](SINGLETON_FUSION.md) mô tả matching hai nguồn và giữ singleton score>=0.7. Kết quả mới 7,631 box, đủ 204 frame/ảnh/video và verification PASS; khôi phục coverage cho nhiều dự đoán score cao. Artifacts được lưu riêng, kết quả KBF bên dưới giữ nguyên.

**Thử nghiệm ba model đã hoàn thành:** [THREE_MODEL_DETECTION.md](THREE_MODEL_DETECTION.md) thêm BEVFusion LiDAR-only. Kết quả fusion cuối8,414 box, thêm783 Vehicle so với hai model, có video sáu ô và verification PASS. PCD intensity toàn0; chưa có GT để khẳng định tăng accuracy.

| Hạng mục | Kết quả |
|---|---|
| PCD | 204 binary XYZI |
| Tần số | xấp xỉ 20 Hz |
| Thời lượng | 10.140826 s |
| Trajectory | 497 pose, xấp xỉ 5 Hz |
| Coverage pose | Bao phủ toàn bộ timestamp PCD |
| Khoảng cách tối đa tới pose gần nhất | xấp xỉ 0.100039 s; cần nội suy |
| Checkpoint | Cả hai đọc được trên CPU, OpenPCDet 0.6.0 |
| Input channels | Lyft = 3; nuScenes = 4 |
| GPU | RTX 5060 Laptop 8 GB |

## Kết quả toàn sequence

Đếm ở score >= 0.3 trên toàn vùng model `[-75.2,75.2] m` theo X/Y. Các số là tổng **box-frame**, không phải số đối tượng duy nhất.

| Output | Vehicle | Pedestrian | Cyclist | Tổng | Trung bình/frame |
|---|---:|---:|---:|---:|---:|
| Lyft PV-RCNN++ | 6,634 | 346 | 1 | 6,981 | 34.22 |
| nuScenes VoxelRCNN | 11,470 | 645 | 23 | 12,138 | 59.50 |
| KBF | 4,619 | 314 | 2 | 4,935 | 24.19 |

Số box/frame min–max: Lyft 23–42, nuScenes 43–73, KBF 14–34.

## Cách xử lý đã thực hiện

1. Chỉ đọc `*.pcd`, bỏ qua `.DS_Store`; đọc timestamp từ header, không lấy `00000` trong tên file làm timestamp.
2. PCD FM → vehicle bằng toàn bộ ma trận `T_vehicle_lidar.FM`, gồm rotation và translation. Không dịch Z thêm lần nữa (`SHIFT_COOR=0`).
3. Nội suy trajectory LiDAR ở timestamp PCD bằng SLERP quaternion và tuyến tính translation. Đổi pose bằng `T_world_vehicle = T_world_lidar @ inverse(T_vehicle_lidar.FM)`.
4. Lyft dùng XYZ một sweep. nuScenes dùng XYZ + time lag thực tế, tối đa 10 sweeps liên tiếp, history lớn nhất 0.450186 s. Những frame đầu dùng history sẵn có, không nhân bản frame đầu.
5. Loại điểm trong ô ego `abs(x)<1.5 AND abs(y)<1.5` ở hệ vehicle, mask range theo config, batch size 1, eval mode, không TTA. Intensity không được đưa vào hai model này.
6. Chạy hai model lần lượt. Điều chỉnh config đầu vào cho đúng checkpoint; mọi tensor trọng số đều được load, không chấp nhận model thiếu trọng số.
7. Quy đổi tên lớp về Vehicle/Pedestrian/Cyclist và gọi KBF có sẵn của repo. Không chạy tracking, temporal refinement hoặc training.

Config độc lập: `runs/two_model_detection/configs/{lyft,nuscenes}.yaml`. Runner: [tools/two_model_detection.py](../../../tools/two_model_detection.py).

## Tham số fusion và giới hạn phương pháp

| Tham số | Vehicle | Pedestrian | Cyclist |
|---|---:|---:|---:|
| Score admission | 0.3 | 0.3 | 0.3 |
| DISCARD | 2 | 2 | 2 |
| Radius tâm 3D (m) | 1.5 | 0.3 | 0.2 |
| NMS threshold | 0.1 | 0.3 | 0.1 |

Trọng số hai nguồn bằng nhau theo lớp; KBF dùng score làm một phần trọng số box. Score của hai detector chưa được calibration theo dữ liệu này.

KBF gốc gom các box gần nhau và yêu cầu ít nhất hai box hỗ trợ; không yêu cầu hai model khác nhau. Đo lại hỗ trợ gần cho từng box KBF cho kết quả:

- 4,176/4,935 box có dự đoán gần từ **cả hai model**.
- 759/4,935 box chỉ có dự đoán gần từ **nuScenes**.
- 0 box chỉ có nguồn gần từ Lyft; 0 box không có nguồn gần nào.

“Hỗ trợ gần” nghĩa là cùng supercategory và khoảng cách tâm 3D <= radius của lớp. Đây không phải matching IoU, không phải số true positives, và không chứng minh đúng đối tượng. Các box đơn lẻ không đủ hỗ trợ có thể bị KBF loại.

## Nhận xét trực quan có giới hạn

Đã xem frame 0, 20, 100 và 203. Nhiều box xe nằm theo các hàng cụm điểm trong cảnh; nuScenes phát hiện dày hơn Lyft. Sau KBF, một phần box trùng/lân cận được hợp nhất và nhiều box chỉ xuất hiện ở một nguồn bị mất, thấy rõ ở các hàng phía Y dương của frame 100/203.

**Chưa đủ bằng chứng để kết luận fusion tốt hơn.** Cấu hình KBF hiện tại giữ box khá bảo thủ: ít box hơn cả hai model. Chưa có GT để biết phần bị loại là false positives hay các xe thật bị bỏ sót. KBF có thể giảm recall; cần xem video/ảnh và gán nhãn một tập frame đại diện nếu muốn kết luận định lượng.

Hai nguồn khác nhau cả kiến trúc, dataset huấn luyện lẫn số sweeps; phép so sánh này kiểm tra ensemble thực tế đang có, không cô lập ảnh hưởng của riêng kiến trúc.

## Runtime và kiểm tra

| Hạng mục | Lyft | nuScenes |
|---|---:|---:|
| Tensor trọng số load | 419/419 | 303/303 |
| Frame hoàn thành | 204/204 | 204/204 |
| Median forward sau frame đầu | 134.08 ms | 113.08 ms |
| Tổng thời gian forward 204 frame | 28.23 s | 24.53 s |
| Peak CUDA allocated | 526.53 MiB | 1753.35 MiB |

Forward được đo có CUDA synchronize, không gồm đọc PCD/cache, ghép sweeps, build model hoặc render ảnh. Không dùng các số trên làm FPS end-to-end. Peak allocated là bộ nhớ do PyTorch theo dõi, không phải tổng VRAM của tiến trình. Không gặp OOM.

- Ba test input/geometry: PASS.
- Phép thử GPU spconv và torch-scatter: PASS.
- Build đủ 5 CUDA extensions với CUDA 12.8/sm_120: PASS, không sửa source lõi OpenPCDet.
- Transform roundtrip float64: sai số max `1.7053e-13 m`; cache float32 sai số max `7.6169e-6 m`.
- Hai đường biến đổi tích lũy sweeps tương đương: sai số max `1.8822e-6 m` trên mẫu kiểm tra.
- `verify`: PASS cho coverage/thứ tự frame, đầy đủ trọng số, box hữu hạn/kích thước dương, đủ ảnh và đếm độc lập khớp summary.
- Video ffprobe: H.264, 1980×660, 204 frame, 20 FPS, 10.2 s. Video dùng FPS cố định gần tần số thật, không biểu diễn chính xác từng khoảng timestamp biến thiên.

## Artifacts để xem lại

- [Video ba cột](../../../runs/two_model_detection/comparison_20fps.mp4)
- [Ảnh frame 0](../../../runs/two_model_detection/comparison/frame_00000.png), [frame 100](../../../runs/two_model_detection/comparison/frame_00100.png), [frame 203](../../../runs/two_model_detection/comparison/frame_00203.png)
- [Summary và số box từng frame](../../../runs/two_model_detection/summary.json)
- [Verification](../../../runs/two_model_detection/verification.json)
- [Manifest và pose từng frame](../../../runs/two_model_detection/manifest.json)
- [Geometry checks](../../../runs/two_model_detection/geometry_checks.json)
- [Input SHA256 fingerprints](../../../runs/two_model_detection/input_fingerprints.json)
- [Environment freeze](../../../runs/two_model_detection/environment.freeze)
- Predictions gốc: `lyft/result.pkl`, `nuscenes/result.pkl`; fusion: `fusion/kbf.pkl`, đều trong `runs/two_model_detection`.
- Logs: `logs/install.log`, `torch_scatter_install.log`, `build.log`, `build_retry1.log`, `prepare.log`, `input_tests.log`, `gpu_environment.log`, `smoke_*.log`, `full_*.log`.

Ảnh hiển thị ±40 m; số box ở tiêu đề/thống kê tính trên toàn range model, kể cả box ngoài khung ảnh. Trục X vehicle hướng trước, Y hướng trái; cyan Vehicle, vàng Pedestrian, hồng Cyclist. Box lưu ở hệ vehicle, không phải LiDAR-top ban đầu.

## Tái lập và trạng thái workspace

Xem [COMMANDS.md](COMMANDS.md). Container riêng `ms3d-fusion-20261105` được giữ lại để chạy lại. PCD/checkpoint/calibration/trajectory gốc không bị chỉnh sửa. `.gitignore` gốc đã có thay đổi từ người dùng và được giữ nguyên; `runs/.gitignore` riêng loại artifacts khỏi Git. Các kết quả nằm trên đĩa local, không được tự động đưa vào Git.

## Smoke test: frame_00020

| Output ở score >= 0.3 | Vehicle | Pedestrian | Cyclist | Tổng |
|---|---:|---:|---:|---:|
| Lyft | 33 | 3 | 0 | 36 |
| nuScenes | 50 | 3 | 0 | 53 |
| KBF | 23 | 2 | 0 | 25 |

- Load trọng số: Lyft 419/419, nuScenes 303/303, không thiếu/thừa key.
- nuScenes dùng 10 sweeps, history thực tế 0.449958 s tại frame này; Lyft dùng 1 sweep.
- Forward lần đầu: Lyft 4.044 s, nuScenes 0.847 s. Có warmup/JIT; không dùng để suy ra throughput ổn định.
- Peak CUDA allocated: Lyft 523 MiB, nuScenes 1594 MiB; số này không phải tổng VRAM báo bởi nvidia-smi.
- KBF có 22 box được hỗ trợ gần bởi cả hai model và 3 box chỉ được hỗ trợ gần bởi nuScenes. Định nghĩa hỗ trợ: cùng supercategory, khoảng cách tâm 3D <= radius theo lớp.
- Đã xem ảnh: nhiều box xe nằm theo các hàng điểm đỗ xe; KBF loại nhiều box so với nuScenes. Chưa có GT để phân biệt giảm false positives và tăng bỏ sót.

Ảnh smoke: `runs/two_model_detection/smoke/comparison/frame_00020.png`.

Smoke artifacts được giữ riêng để đối chiếu với full sequence. Các số runtime smoke chứa warmup/JIT, không dùng thay số đo full run phía trên.
