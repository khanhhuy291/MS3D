# BEVFusion + CenterPoint + VoxelRCNN → MS3D++ — 2026-11-16

Trạng thái: **HOÀN THÀNH** 204/204frame; fusion, tracking, refinement và kiểm tra độc lập PASS. Output riêng: `runs/fusion_bcv` và `runs/ms3d_bcv_round1`. Không ghi đè fusion/pseudo-label bộ ba cũ có Lyft.

## Input và phương pháp

- VoxelRCNN: `runs/two_model_detection/nuscenes/result.pkl` (10sweep ở các frame đủ lịch sử).
- CenterPoint: JSON trong `runs/centerpoint_voxel1440_fm_singleframe/predictions` (1sweep). Đổi model_lidar→vehicle bằng Z+1.8m theo metadata; JSON yaw đã length-axis, gravity center.
- BEVFusion: `runs/three_model_detection/bevfusion/result.pkl`, đã chuẩn hóa corners/yaw/bottom-center sang vehicle ở lần chạy trước.
- Ba nguồn có204frame. Dùng pose/timestamp/prepared vehicle cloud đã xác minh. Không chạy lại detector.
- Source order: `[voxelrcnn,centerpoint,bevfusion]`. Hungarian seed VoxelRCNN/CenterPoint, sau đó BEVFusion được ghép vào nhóm khi phù hợp mọi thành viên.
- Admission0.1; BEV IoU gate Vehicle0.3/Pedestrian0.2/Cyclist0.2; XY gate2m/0.75m/1m; Z gate1m. Nhóm2/3nguồn score>=0.3, singleton>=0.7. Không có cam kết giữ mọi singleton sau NMS/refinement.
- Fusion giữ center/yaw của thành viên score mạnh nhất, trung bình score-weighted dimensions/score; căn chỉnh trục tương đương cho mọi pair/triple. NMS fusion0.5/0.3/0.3.
- Giữ3superclasses Vehicle/Pedestrian/Cyclist; barrier/traffic_cone không tham gia MS3D fusion, bản JSON gốc vẫn nguyên.

## Tracking/refinement

Chạy native MS3D++ cho veh_all, veh_static, pedestrian, rồi lọc track/motion/merge/rolling KBF15quan sát/propagation±10frame/update_ps/NMS/points/positive thresholds. Cấu hình customRound1, timestamp thực20Hz; các ngưỡng đếm frame giữ nguyên để so baseline.

Giới hạn native: update_ps chỉ xuất Vehicle/Pedestrian, select_ps_by_th có mask Cyclist trỏ nhầm class2. Lưu kết quả native riêng `final_native_two_class.pkl`. Bản cuối `final_ps_dict.pkl` bổ sung Cyclist từ fusion qua NMS0.3, num_pts>1, ego filter, POS_TH0.5; **Cyclist không có temporal tracking/refinement**. Vehicle/Pedestrian giữ đúng native, kể cả điều kiện pedestrian export khi initial frame trống đã được chẩn đoán trước; lần này chưa áp dụng bản sửa flickering.

## Lệnh

```bash
docker start ms3d-fusion-20261105
docker exec ms3d-fusion-20261105 python tools/run_bcv_ms3d.py
```

Có thể chạy riêng `--phase fusion`, `--phase track`, `--phase refine`. File input/output và SHA256 lưu theo experiment. Các kết quả/kiểm tra thực tế sẽ bổ sung khi xong.


## Kết quả thực thi

### Fusion và nhãn cuối

| Artifact/tập nhãn | Vehicle | Pedestrian | Cyclist | Tổng |
|---|---:|---:|---:|---:|
| Fusion BCV |9,184|302|36|9,522|
| Fusion nếu áp POS_TH0.7/0.6/0.5 trực tiếp |4,335|94|5|4,434|
| Final native chỉ V/P, gồm ignore |9,258|304|0|9,562|
| Final bổ sung Cyclist, gồm ignore |9,258|304|36|9,598|
| Final positive-only |6,084|156|5|6,245|
| Ignore trong final |3,174|148|31|3,353|

Final dương tăng1,811 so với áp cùng POS_TH lên fusion đầu vào, gồmVehicle+1,749/Pedestrian+62/Cyclist+0. Tracking và rolling KBF có nâng score; con số này không phải số vật thể mới hoặc accuracy gain.

Native2class có6,240positive và3,322ignore. Bản bổ sung Cyclist chỉ thêm36box fusion đã qua lọc, gồm5positive/31ignore; Vehicle/Pedestrian khớp **chính xác từng hàng và metadata** với native. Không gắn nhãn rằng Cyclist đã được temporal refine.

### Nguồn đóng góp sau fusion

| Nhóm nguồn | Box |
|---|---:|
| VoxelRCNN + CenterPoint |7,640|
| VoxelRCNN + CenterPoint + BEVFusion |890|
| VoxelRCNN + BEVFusion |42|
| CenterPoint + BEVFusion |359|
| Singleton VoxelRCNN |350|
| Singleton CenterPoint |79|
| Singleton BEVFusion |162|

Tổng seed-pairs trước score/NMS11,454; BEVFusion ghép vào1,092nhóm2nguồn và1,247nhóm1nguồn. Sau score còn10,760candidate; NMS fusion loại1,238, còn9,522box. Các số trước score không cộng trực tiếp với provenance cuối.

CenterPoint có47,020box gốc; loại khỏi MS3D fusion497traffic_cone và147barrier theo3superclasses, còn46,376box đưa qua admission/matching. Tất cả dữ liệu JSON nguồn và viewer raw vẫn giữ đủ lớp.

### Tracking/refinement

| Nhánh | Track thô | Qua confidence | Qua length | Sau merge/refine |
|---|---:|---:|---:|---:|
| veh_all |260|113|không lọc riêng|60|
| veh_static |386|59|54|54|
| pedestrian |21|2|2|2|

Tracking mất khoảng47.97s/28.46s/1.49s cho ba nhánh. Sau merge: veh_all59static+1dynamic; static branch53static+1dynamic; ped2dynamic. Không có GT xác nhận motion state.

veh_all sau merge có4,495hàng lịch sử nhưng4,491frame-keyed box:4hàng trùng frame được resolved trong frame mapping native. Một lịch sử track có206hàng trong sequence204frame vì duplicate frame; không phải206timestamp khác nhau. Output theo frame của MS3D dùng mapping đã resolved.

Static có5,682quan sát/rolling box; propagation tổng6,185track-frame proposal, gồm503vị trí ngoài biên.53static track được đưa vào nhánh static update;1dynamic track không được dùng ở nhánh static. Các proposal có thể trùng nhánh veh_all/detection gốc.

### Ghép/lọc cuối

| Bước | Số box/proposal |
|---|---:|
| Tổng proposal trước final NMS, gồm nhánh Cyclist |20,329|
| NMS loại proposal chồng |10,665|
| Sau NMS, đưa vào đếm điểm |9,664|
| Loại num_pts<=1 |66|
| Loại ego-center |0|
| Giữ trong final native + Cyclist |9,598|
| Đánh ignore theo POS_TH |3,353|
| Positive-only |6,245|

Có69singleton proposal đi qua nhánh bypass NMS, nên `summary.json` ghi nms_candidates20,260/nms_retained9,595 cho các lần gọi NMS mảng>1box; tổng tương ứng20,329/9,664 đã gồm69bypass. Cyclist36box đều đủ point support và qua ego filter;31box bịignore bởi POS_TH0.5.

## Kiểm tra đã hoàn thành

- 12 test fusion/geometry và5test bridge PASS. Các test legacy vẫn PASS sau khi thêm source names và axis alignment cho BCV.
- [Fusion verification](../../../runs/fusion_bcv/verification.json) PASS:204frame; biến đổi46,376box CenterPoint kiểm tra độc lập; source-index uniqueness; mọi cặp trong nhóm qua gate; center/yaw/size/score tính lại độc lập; singleton threshold và NMS geometry đúng.
- [Final verification](../../../runs/ms3d_bcv_round1/verification.json) PASS trên9,598box: số điểm tính lại độc lập khớp100%, dimensions/finite, class sign, positive subset, ego removal, propagation<=10frame.
- Chuẩn hóa373box Vehicle DX/DY/yaw trước tracking bảo toàn footprint; minBEVIoU=0.9999884, center/score không đổi.
- [Cyclist verification](../../../runs/ms3d_bcv_round1/cyclist_verification.json) PASS:36box Cyclist khớp fusion sau point/ego/score filters; Vehicle/Pedestrian chính xác như native2class.
- SHA256 sources và baseline fusion/final cũ không đổi.30fingerprint source native trùng lần chạy cũ; không sửa thuật toán core. Wrapper có nhánh bảo toàn Cyclist được nêu riêng.
- 8 test viewer data-contract PASS;12nhóm Chrome/WebGL regression PASS với11case, gồm hai case BCV, overlay, filters, playback, export và responsive.

## So sánh và giới hạn

| Bộ ba | Fusion V/P/C | Positive cuối V/P/C | Tổng positive |
|---|---|---|---:|
| Lyft + VoxelRCNN + BEVFusion trước |8,057/357/0|6,881/324/0|7,205|
| VoxelRCNN + CenterPoint + BEVFusion mới |9,184/302/36|6,084/156/5|6,245|

Bộ mới có nhiều fusion box hơn nhưng ít positive cuối hơn (6,245so7,205). Khác score distribution, nguồn detection và track đủ confidence; chưa có GT thì không suy ra chất lượng tăng/giảm từ bảng này. So trên common ROI/nhãn GT mới kết luận accuracy.

[output_changes.json](../../../runs/ms3d_bcv_round1/output_changes.json):6,056positive có correspondence với fusion input cùng lớp bằng Hungarian BEVIoU>=0.1;189không ghép được theo tiêu chí đó. Dịch tâm giữa các cặp: mean0.0759m, median0.0382m, P95=0.2335m, max1.3571m. Ưu tiên xemframe133/49/138, chủ yếu thay đổi Z>1.28m. Đây là so prediction trước–sau, chưa phải lỗi soGT; correspondence có thể khớp nhầm vật thể.

### Flickering vẫn còn

[flicker_diagnosis.json](../../../runs/ms3d_bcv_round1/flicker_diagnosis.json):6,134/6,245positive có kept-track support BEVIoU>=0.5,111không có (Cyclist không có temporal track cũng thuộc nhóm này). Có61lần single-frame geometric island theo ghép Hungarian world BEVIoU>=0.3;60không có kept-track support. Số lần không phải số vật thể.

veh_all4,482/4,491slot có positive coverage;9slot chỉ được phủ bởi ignore ởIoU0.3, ởIoU0.5 không có correspondence. Chúng không tạo lỗ nội bộ trong khoảng positive trước/sau của track theo phép đo.53static track có6,161/6,161slot được positive phủ, không có lỗ nội bộ theoIoU0.3/0.5.

Ped có155/158slot được positive phủ.3slot mất do native initial-pedestrian-empty guard; trong đó **frame165**, track13, nằm giữa frame164và166 có nhãn. Đã giữ baseline native để so, chưa áp dụng sửa lỗi xuất người hoặc hysteresis/longer-age. Kết quả mới chưa giải quyết triệt để flickering.

## File và lệnh sử dụng

- [Fusion mới](../../../runs/fusion_bcv/fusion.pkl), [summary fusion](../../../runs/fusion_bcv/summary.json).
- [Final native2class](../../../runs/ms3d_bcv_round1/final_native_two_class.pkl), [native positive-only](../../../runs/ms3d_bcv_round1/final_native_positive_only.pkl).
- [Final đầy đủ có Cyclist](../../../runs/ms3d_bcv_round1/final_ps_dict.pkl), [positive-only](../../../runs/ms3d_bcv_round1/final_positive_only.pkl).
- [summary](../../../runs/ms3d_bcv_round1/summary.json), [refinement audit](../../../runs/ms3d_bcv_round1/refinement_audit.json), [cyclist audit](../../../runs/ms3d_bcv_round1/cyclist_output_audit.json).
- `tracks_world_veh_all.pkl`, `tracks_world_veh_static.pkl`, `tracks_world_ped.pkl`: track thô world; `stages/`: snapshots motion/merge/rolling/propagation/preselection.
- [Fusion log](../../../runs/ms3d_bcv_round1/logs/fusion.log), [tracking log](../../../runs/ms3d_bcv_round1/logs/tracking.log), [refinement log](../../../runs/ms3d_bcv_round1/logs/refinement.log), [verification log](../../../runs/ms3d_bcv_round1/logs/verification.log).
- [Native fingerprints](../../../runs/ms3d_bcv_round1/code_fingerprints.json), [pipeline fingerprints](../../../runs/ms3d_bcv_round1/pipeline_code_fingerprints.json), [baseline hashes](../../../runs/fusion_bcv/baseline_sha256.json).

```bash
# Full pipeline; identical validated input/config may reuse saved raw tracks.
docker exec ms3d-fusion-20261105 python tools/run_bcv_ms3d.py

# Independent verification, including all final point counts:
docker exec ms3d-fusion-20261105 python tools/verify_bcv_ms3d.py

# Data-contract checks for all viewer cases:
python3 tools/test_local_viewer.py
```

Viewer thêm `fusion_bcv` và `ms3d_bcv`. Khởi động lại server cũ để nạp catalog mới:

```bash
cd /home/khanhhuy/MS3D
python3 tools/serve_local_viewer.py --port 8765
```

Mở `http://127.0.0.1:8765/?frame=100&case=ms3d_bcv&score=0.3`, bật overlay `fusion_bcv` để so trước–sau, hoặc `ms3d_final` để so kết quả cũ có Lyft. Giữ filter nguồn Tất cả vì final không còn provenance từng box. Score slider thấp không phục hồi ignore đã loại khỏi positive-only.

QA/ảnh riêng tại `runs/local_viewer/bcv_qa`. Server QA8767/Chrome tạm dừng sau kiểm tra, không dừng viewer có sẵn của người dùng. Không train detector hoặc chạy round2 trong lượt này.


## Config và trace source

Cấu hình đã chạy: [ps_config.yaml](../../../runs/ms3d_bcv_round1/ps_config.yaml).

| Nhánh tracking | Score chính | Association | Max age | Confidence filter |
|---|---:|---|---:|---|
| veh_all |0.7|GIoU|2frame|>=4quan sát score>0.7|
| veh_static |0.5|IoU2D|3frame|>=5quan sát score>0.7 và>=10box|
| pedestrian |0.3|GIoU|2frame|>=2quan sát score>0.6 và>=10box; bỏ track static|

KF dùng dt timestamp thực; min hits birth2. Final NMS Vehicle0.05/Pedestrian0.5, Cyclist passthrough0.3. Đếm điểm current single sweep, native margin XY0.1m, giữnum_pts>1. Ego filter abs(x)<1m AND abs(y)<1m. POS_TH0.7/0.6/0.5; ignore class_id âm. Box schema `[x,y,z,dx,dy,dz,yaw,class_id,score]`, tâm hình học trong vehicle. World tracking dùng `T_world_vehicle=T_world_FM@inverse(T_vehicle_FM)`, SHIFT_COOR=0.

- [Runner BCV](../../../tools/run_bcv_ms3d.py): input adapter, fusion và cấu hình bridge output riêng.
- [Fusion source-aware](../../../tools/fuse_three_models.py): `fuse_frame` / `fuse_members`; tham số align_all_pairs chỉ bật trong config BCV, mặc định legacy giữ nguyên.
- [Bridge](../../../tools/run_ms3d_from_three_model_fusion.py): `prepare`, `track`, `refine`, snapshot/audit và nhánh bảo toàn Cyclist.
- [Native tracker](../../../pcdet/utils/tracker_utils.py): `get_tracklets`; [native refinement](../../../pcdet/utils/ms3d_utils.py): `refine_veh_labels`, `refine_ped_labels`, `update_ps`, `select_ps_by_th`.
- [Verifier BCV](../../../tools/verify_bcv_ms3d.py) và [verifier pseudo-label](../../../tools/verify_ms3d_pseudo_labels.py): độc lập với phép tính geometry/point count trong pipeline.

## Ảnh đối chiếu đã kiểm tra

[QA comparison JSON](../../../runs/local_viewer/bcv_qa/bcv_comparison_verification.json) PASS trên5trường hợp trực tiếp, ngoài12nhóm regression. Frame100 fusionBCV49box, final33positive; overlay so đúng số box API.

- [Frame100: fusion mới so fusion cũ](../../../runs/local_viewer/bcv_qa/frame_00100_fusion_bcv_vs_fusion3.png).
- [Frame100: final mới so fusion mới](../../../runs/local_viewer/bcv_qa/frame_00100_ms3d_bcv_vs_fusion_bcv.png).
- [Frame100: final mới so final cũ](../../../runs/local_viewer/bcv_qa/frame_00100_ms3d_bcv_vs_ms3d_final.png).
- [Frame133: final/fusion, outlier Z](../../../runs/local_viewer/bcv_qa/frame_00133_ms3d_bcv_vs_fusion_bcv.png).
- [Frame165: final/fusion, pedestrian export gap](../../../runs/local_viewer/bcv_qa/frame_00165_ms3d_bcv_vs_fusion_bcv.png).
