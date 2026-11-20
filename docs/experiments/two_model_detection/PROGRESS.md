# Tiến độ — 2026-11-05

## Đã hoàn thành

- Kiểm tra 204 PCD, header, tần số, coverage của trajectory và checkpoint.
- Người dùng xác nhận hệ LiDAR FM và pose LiDAR.
- Kiểm tra GPU ngoài sandbox: RTX 5060 Laptop 8 GB, driver 595.99.02.
- Tạo container riêng `ms3d-fusion-20261105` từ `bevfusion:rtx5060`; không thay đổi các container BEVFusion có sẵn.
- Container dùng PyTorch 2.7.1+cu128, CUDA toolkit 12.8, có kiến trúc sm_120.
- Kiểm tra package index: không có `spconv-cu128`; có `spconv-cu126==2.3.8`.

## Cập nhật: chuẩn bị dữ liệu và GPU operators

- Đã chuẩn bị 204 frame trong `runs/two_model_detection/prepared`, không sửa PCD gốc.
- Đã tạo hai config riêng trong `runs/two_model_detection/configs`.
- Đã cài `spconv-cu126==2.3.8`, `torch-scatter==2.1.2+pt27cu128`, Open3D và dependencies.
- Phép thử GPU spconv: output `(2,8)`, toàn bộ giá trị hữu hạn; scatter_mean cho kết quả đúng.
- Ba test input/geometry đều pass: timestamp header, từ chối layout sai, biến đổi tương đương giữa hai hệ.
- Sai số phép chuyển LiDAR→vehicle→LiDAR float64 tối đa: `1.7053e-13 m`.
- Sai số cache float32 so với transform float64: `7.6169e-6 m`.
- Sai số giữa hai đường tích lũy sweeps tương đương: `1.8822e-6 m`.
- Nhóm điểm frame đầu trong khoảng XY 5–25 m và `abs(z)<0.5`: median Z `-0.00497 m` (chỉ là kiểm tra hình học giới hạn).
- Lần build đầu bị lỗi Git ownership dẫn tới version rỗng; đã đặt safe.directory **bên trong container** và build lại. Log lần đầu vẫn được giữ.

## Cập nhật: smoke inference thành công

- Build đủ 5 CUDA extensions thành công với CUDA 12.8/sm_120. Không sửa code lõi OpenPCDet để build.
- Lyft load đủ 419/419 tensors; nuScenes load đủ 303/303 tensors; không missing/unexpected key.
- Smoke frame `frame_00020`: Lyft 42 box score>=0.1, nuScenes 134 box score>=0.1.
- Ở score>=0.3: Lyft 36 (33 vehicle, 3 pedestrian), nuScenes 53 (50 vehicle, 3 pedestrian), KBF 25 (23 vehicle, 2 pedestrian).
- KBF smoke: 22 box có tâm gần dự đoán từ cả hai nguồn; 3 chỉ có hỗ trợ gần từ nuScenes theo bán kính KBF. Đây không phải xác nhận đúng/sai bằng GT.
- Đã xem ảnh BEV smoke. KBF loại nhiều detection đơn lẻ, cần đánh giá cả bỏ sót và giảm false positives.
- Đã lưu bản smoke tại `runs/two_model_detection/smoke` trước khi full run.

## Cập nhật: full inference và KBF hoàn thành

- Cả hai model đã chạy đủ 204/204 frame, không gặp OOM.
- Output score>=0.3 trên toàn sequence (box-frame, không phải object duy nhất):
  - Lyft: 6,634 Vehicle + 346 Pedestrian + 1 Cyclist = 6,981.
  - nuScenes: 11,470 Vehicle + 645 Pedestrian + 23 Cyclist = 12,138.
  - KBF: 4,619 Vehicle + 314 Pedestrian + 2 Cyclist = 4,935.
- Đã kiểm tra box KBF hữu hạn và kích thước dương ở toàn bộ 204 frame.
- Median forward sau frame đầu: Lyft 134.08 ms; nuScenes 113.08 ms (không gồm đọc/ghép point cloud).
- Peak CUDA allocated: Lyft 526.53 MiB; nuScenes 1753.35 MiB.

## Cập nhật cuối: hoàn thành

- Đã render đủ 204 ảnh BEV so sánh.
- Đã tạo video H.264 1980×660, 204 frame, 20 FPS, 10.2 s; ffprobe xác nhận metadata.
- KBF support gần: 4,176 box có cả hai model, 759 box chỉ gần nuScenes theo radius lớp.
- Kiểm tra artifact độc lập: PASS; frame IDs/thứ tự/coverage, checkpoint completeness, box hợp lệ, ảnh và tổng số box khớp summary.
- Đã xem frame 0/20/100/203; REPORT.md ghi rõ nhận xét giới hạn và chưa kết luận accuracy khi không có GT.
- Đã cập nhật REPORT.md và COMMANDS.md để xem lại/tái lập.

## Bước chưa thực hiện, ngoài phạm vi lần thử này

- Chưa đo AP/precision/recall vì chưa có GT.
- Chưa tuning KBF, tracking, temporal refinement hoặc training.

## Lưu ý trạng thái ban đầu

`.gitignore` đã có thay đổi của người dùng trước khi bắt đầu; không chỉnh sửa hoặc hoàn tác thay đổi đó.

## Chẩn đoán theo phản hồi mất nhiều box — 2026-11-05

- Thêm script audit từng khâu trên predictions đã lưu; baseline tái dựng khớp kết quả gốc ở toàn bộ 204 frame.
- 19,119 box đầu vào, 8,622 không vào nhóm, 4,994 ứng viên KBF, 59 bị NMS loại, 4,935 đầu ra.
- nuScenes có 864 box score>=0.9 không được dùng trong nhóm vì thiếu hỗ trợ.
- Thử fusion-only: NMS xe 0.5 cho 4,953 box; admission 0.1 cho 6,656 box; cả hai cho 6,790 box (đầu ra luôn score>=0.3).
- Ghi thêm lỗi source `boxes.shape == 2` khiến nhánh special-case không chạy; chưa sửa hoặc đo hiệu quả của việc sửa.
- Chi tiết và giới hạn: [KBF_DIAGNOSIS.md](KBF_DIAGNOSIS.md). Giữ nguyên predictions, video, config và code KBF gốc.

## Thử biến thể giữ singleton — 2026-11-05

- Người dùng yêu cầu thực hiện hướng fusion giữ box đơn score cao.
- Đã viết runner riêng và 8 test cho assignment, spatial gate, yaw, pair fusion và singleton.
- Chạy admission=0.1, pair output>=0.3, singleton>=0.7; matching khác nguồn qua BEV IoU và khoảng cách; NMS Vehicle=0.5.
- Đã hoàn thành 204/204 frame, 204 ảnh và video tại `runs/two_model_detection_singletons`.
- 8 test PASS; verification độc lập PASS gồm provenance/matching/geometry, tất cả PNG và SHA256 input gốc không đổi.
- Fusion mới: 7,274 Vehicle + 357 Pedestrian + 0 Cyclist = 7,631 box; 4,865 matched, 2,598 singleton nuScenes, 168 singleton Lyft.
- Coverage của predictions gốc score>=0.7 theo cùng lớp/BEV IoU>=0.5: khôi phục thêm 255 Lyft và 2,338 nuScenes so với KBF gốc. Không phải recall theo GT.
- Ví dụ nuScenes score0.96 ở frame100 đã được giữ đúng box gốc.
- Thử thêm singleton>=0.5, không render, output riêng: 9,858 box. Chưa chọn cấu hình phụ vì chưa có GT.
- Video đã kiểm tra: H.264 2640×660, 204 frame, 20FPS, 10.2s. Đã đổi owner riêng thư mục kết quả mới về 1000:1000 để host có thể tạo video.
- Chi tiết phương pháp/trạng thái: [SINGLETON_FUSION.md](SINGLETON_FUSION.md).

## Mở rộng ba detector — 2026-11-05

- Người dùng thêm `lidar-only-det.pth` và yêu cầu thử ba detector/fusion.
- Đã đọc checkpoint (306 tensor), kiểm tra config/source BEVFusion và tạo container riêng, source BEVFusion mount read-only.
- Đang smoke inference BEVFusion với input LiDAR-top gốc, quy đổi output dựa trên corners sang vehicle.
- Báo cáo riêng: [THREE_MODEL_DETECTION.md](THREE_MODEL_DETECTION.md).
- Smoke đầu thiếu CUDA extension ở source host read-only; đã dùng source build sẵn của image `/workspace/bevfusion-adapter`, không sửa source host.
- BEVFusion full 204/204 frame hoàn thành, load306/306, geometry self-test PASS; 7 test fusion/geometry mới PASS.
- Phát hiện intensity ở toàn bộ PCD bằng0; giữ nguyên input và ghi rõ domain gap với BEVFusion.
- Đang render fusion ba nguồn, kiểm tra baseline hai model khớp trên từng frame.
- Đã phát hiện cần căn chỉnh biểu diễn DX/DY/yaw giữa nguồn BEVFusion và OpenPCDet trước khi trung bình kích thước. Đã bổ sung hai test, tổng 9 test PASS; lưu bản trước sửa riêng và đang xuất lại output cuối.
- Hoàn thành: 204 ảnh/video sáu ô và fusion cuối 8,057 Vehicle +357 Pedestrian +0 Cyclist =8,414, thêm783 Vehicle so với fusion hai model.
- Verification độc lập PASS: geometry/corners/upright enclosure, axis alignment, gate mọi cặp, source uniqueness, framecoverage, boxvalid, ảnh204, SHA256 gốc không đổi; baseline hai model tái lập chính xác.
- Coverage predictions score>=0.7 theo BEV IoU>=0.5: BEVFusion mới được phủ125 prediction, nuScenes giảm1, Lyft giữ nguyên. Không phải recall theo GT.
- Video đã đọc lại bằng ffprobe: H.2641980×1320, 204frame,20FPS,10.2s. Báo cáo THREE_MODEL_DETECTION.md đã cập nhật số liệu full/commonrange, phương pháp, lệnh và giới hạn.

## Web visualization local — 2026-11-05

- Người dùng yêu cầu web local để xem đồng thời BEV/3D và các trường hợp.
- Đã tạo viewer Python stdlib+NumPy, Three.js r160/OrbitControls vendored để chạy offline. Local-only, không đăng ký/publish site.
- Nạp7case/204frame từ artifacts hiện có, có provenance, bộ lọc score/lớp/số nguồn/nguồnđónggóp, overlay, chọnbox, camera/point controls, exportPNG/JSON.
- 5 test data/API-contract PASS; 12 nhóm QA Chrome thật PASS, có WebGL thực qua SwiftShader, pointer selection/orbit, filter, playback, export JSON/PNG, responsive1024/390.
- Frame100/fusion3 mặc định có44 box/59,352 điểm sausampling+Zclip; API và browser khớp số box. Không có JS runtime/console error hoặc request asset/data ngoài localhost.
- Đã lưu screenshot desktop/mobile và browser_verification.json tại `runs/local_viewer/qa`.
- Đã chỉnh camera Top Z-up và bố cục timeline/FPS qua kiểm tra thực. Server/Chrome QA tạm được dừng sau kiểm tra; lệnh chạy lại là `python3 tools/serve_local_viewer.py --port 8765`.
- Hướng dẫn và cập nhật QA: [LOCAL_VIEWER.md](LOCAL_VIEWER.md).

## Chạy phần còn lại MS3D++ — 2026-11-06

- Người dùng yêu cầu lấy fusion ba model singleton0.7 và chạy tracking/refinement đến pseudo-label cuối.
- Xác nhận input204frame/8,414box, Vehicle8057/Pedestrian357/Cyclist0; singletonkhông có score bằng đúng0.7.
- Đã tạo data bridge giữ vehicle frame/pose/timestamp giây, chuẩn hóa388 Vehicle DX/DY/yaw theo footprint tương đương.
- Chạy native get_tracklets cho3config, rồi refine_veh_labels/refine_ped_labels/update_ps/select_ps_by_th; snapshot/audit riêng trong `runs/ms3d_three_model_round1`.
- Tham số dùng nguyên customRound1; đã hoàn thành và verification PASS. Báo cáo: [FINAL_PSEUDO_LABELS.md](FINAL_PSEUDO_LABELS.md).

- Tracking thô363veh_all/462veh_static/13ped; sau lọc/refinement126/86/3track. Rolling5,848box; propagation thêm1,136proposal ngoài track.
- Ghép21,165proposal -> NMS9,095 -> point support loại38 -> native final9,057. Nhãn dương7,205 (V6,881/P324); ignore1,852 (V1,806/P46).
- Kiểm tra độc lập9,057box PASS: point count khớp100%, dấu class/pose/footprint/propagation/source hashes hợp lệ. 5 test bridge và6 test viewer PASS.
- Final lưu riêng tại `runs/ms3d_three_model_round1/final_ps_dict.pkl`; bản chỉ class dương `final_positive_only.pkl`.
- Đã thêm case `ms3d_final` vào viewer để so với `fusion3`. Chi tiết ngưỡng, data flow, audit và outlier frame123/115/197/201 trong FINAL_PSEUDO_LABELS.md. Tăng nhãn dương1,640 không phải kết luận tăng accuracy.
- 12 nhóm Chrome/WebGL regression PASS với8case; thêm ảnh final-vs-fusion ở115/123/197/201 và review JSON. QA lưu riêng `runs/ms3d_three_model_round1/viewer_qa`; giữ QA và server8765 có sẵn.


## Chẩn đoán flickering — 2026-11-07

- Chưa sửa nhãn/core. Đã phân tích final positive so với retained tracks và frame trước/sau trong world.
- 7,020/7,205positive có kept-track support BEV IoU>=0.5;185không có.104lần xuất hiện đơn frame theo Hungarian world BEV IoU>=0.3, đều thuộc nhóm không có kept-track support.
- Các slot veh_all5,534 và static-active6,893 đều có positive coverage ở IoU0.3/0.5; chưa thấy lỗ hình học trong khoảng track được xuất theo tiêu chí này.
- Xác định2pedestrian slots mất ở frame7 (track0/1) do update_ps chỉ thêm track khi frame có initial pedestrian.
- Ví dụ vùng vehicle(-0.65,31.5) ở frame3: final score0.69351 tại frame2 bịignore,0.74002 frame3 hiện,0.65513 frame4 bịignore; sau đó input mất nhiều frame. Box gốc không có temporal support vẫn gâyflicker.
- Report: [FLICKERING_DIAGNOSIS.md](FLICKERING_DIAGNOSIS.md). Audit/script riêng; baseline và kết quả trước giữ nguyên.


## Kiểm tra khả năng chạy CenterPoint — 2026-11-16

- Repo có detector CenterPoint đăng ký sẵn và config KITTI `uda_centerpoint.yaml` (Car/XYZ/MeanVFE/VoxelResBackBone8x/CenterHead).
- Model nuScenes đã chạy là VoxelRCNN+CenterHead+ROI_HEAD, khác baseline CenterPoint độc lập.
- `checkpoints/` chưa có cặp CenterPoint checkpoint/config độc lập được xác minh. Chưa chạy inference hay sửa input/nhãn cũ.
- Cần checkpoint/config matching trước smoke test; report: [CENTERPOINT_READINESS.md](CENTERPOINT_READINESS.md).


## Đưa detection CenterPoint lên web — 2026-11-16

- Người dùng cung cấp run `centerpoint_voxel1440_fm_singleframe`; đã đọc metadata/204prediction JSON và source runner CenterPoint để xác minh box contract.
- Viewer nạp trực tiếp JSON, đổi model_lidar→vehicle bằng Z+1.8m, giữ score/dim/yaw/original names. Frame IDs/timestamps/point counts khớp prepared dataset.
- Case thứ9 `centerpoint`; đủ47,020box nguồn,11,156box ởscore0.3. Thêm Barrier/Traffic cone, source filter và original class inspector/export.
- 7 test data-contract PASS;12 nhóm Chrome/WebGL regression và8 kiểm tra riêng CenterPoint PASS. Frame100=53box, overlayfusion3=44box; barrierframe199=1box.
- SHA256204JSON+summary nguồn không đổi. QA/screenshot lưu riêng `runs/local_viewer/centerpoint_qa`; không sửa detector/fusion/pseudo-label artifacts.
- Report và lệnh mở: [CENTERPOINT_VIEWER.md](CENTERPOINT_VIEWER.md). Server QA8766/Chrome tạm được dừng sau kiểm tra.


## Fusion bộ ba BCV + MS3D++ — 2026-11-16

- Người dùng yêu cầu BEVFusion + CenterPoint + VoxelRCNN; tạo output riêng `fusion_bcv`, `ms3d_bcv_round1`.
- Đã xác nhận frame coverage và coordinate contract, khởi động lại môi trường MS3D CUDA đã dùng trước.
- Chạy fusion singleton0.7/group0.3, tracking/refinement customRound1. Native refine V/P, giữ Cyclist với spatial/score filters trong output cuối và lưu native2class riêng.
- Đang thực thi; chưa kết luận số liệu. Report: [BCV_MS3D.md](BCV_MS3D.md).

- Hoàn thành204/204frame: fusion9,522box (V9,184/P302/C36); final9,598, positive6,245 (V6,084/P156/C5), ignore3,353. Native2class9,562box/6,240positive lưu riêng.
- Tracking260/386/21track thô; sau lọc/merge/refine60/54/2. Rolling5,682box, propagation thêm503proposal ngoài biên. Ghép20,329proposal -> NMS9,664 -> lọc66box thiếu điểm ->9,598.
- Fusion verification PASS, final kiểm tra9,598point counts khớp100%, Cyclist preservation/native V/P equality PASS.12test fusion,5test bridge,8test viewer,12nhóm Chrome regression PASS. Baseline/source hashes không đổi;30native source fingerprints khớp trước.
- Thêm viewer cases `fusion_bcv`, `ms3d_bcv`. Đo flicker61single-frame geometric islands, còn pedestrian gap frame165 do native export guard; chưa tuyên bố accuracy/flicker đã được sửa. Chi tiết trong BCV_MS3D.md.
