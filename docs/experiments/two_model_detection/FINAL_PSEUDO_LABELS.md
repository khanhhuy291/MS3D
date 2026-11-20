# MS3D++ từ fusion ba model đến pseudo-label cuối

Ngày chạy: 2026-11-06. Trạng thái: **HOÀN THÀNH** 204/204 frame; pipeline và kiểm tra độc lập PASS.

## Input đã xác nhận

- `runs/three_model_detection/fusion.pkl`: 204 frame, 8,414 box =8,057 Vehicle +357 Pedestrian, không có Cyclist.
- Singleton của bộ này dùng ngưỡng0.7, nhóm2/3nguồn dùng0.3. Không lọc toàn bộ input về score>0.7. Không có singleton bằng đúng0.7 trong dữ liệu hiện tại nên cách viết > hay >= không đổi tập input.
- Center và point cloud ở hệ vehicle, Z tâm, mét. Pose từ manifest là `T_world_vehicle = T_world_lidar @ inverse(T_vehicle_lidar.FM)`. SHIFT_COOR=0, không cộng Z hoặc áp dụng extrinsic thêm lần nữa.
- Timestamp giây thực trong manifest, median~0.05s (20Hz). Dùng cùng một sequence liên tục.

## Bridge và chuẩn hóa trước tracking

Runner [run_ms3d_from_three_model_fusion.py](../../../tools/run_ms3d_from_three_model_fusion.py) tạo `ManifestDataset`, đúng contract mà `compatibility_utils` của MS3D dùng: infos, pose, timestamp, sequence, frame index, get_lidar. Point cloud đọc từ prepared `.npy` một sweep. Không dùng loader mặc định vốn suy timestamp từ tên file và giả định0.1s.

388 Vehicle box có DX<DY; swap DX/DY, yaw+=pi/2 và wrap yaw về[-pi,pi). Footprint và Z/center giữ nguyên. Mục đích là cùng một trục dài cho KF và rolling KBF khi nguồn score mạnh thay đổi giữa các frame. Lưu original fusion riêng và chỉ số box đã swap.

`initial_pseudo_labels.pkl` chỉ chứa `gt_boxes (N,9)` để tránh giữ source_indices/provenance cũ không còn đúng sau thêm/xóa/thay box. Provenance fusion gốc vẫn có trong `input_fusion3_original.pkl` và file nguồn không bị sửa.

## Các hàm native được gọi

1. `tracker_utils.prepare_track_cfg` và `get_tracklets` cho veh_all, veh_static, pedestrian.
2. `ms3d_utils.refine_veh_labels`: lọc track, xác định motion, đối chiếu hai lượt tracking, merge đoạn track, rolling KDE/KBF, propagation.
3. `ms3d_utils.refine_ped_labels`: lọc track người theo confidence/độ dài/chuyển động.
4. `ms3d_utils.update_ps`: ghép nhãn ban đầu với track/refined box, NMS, lọc point support và ego box.
5. `ms3d_utils.select_ps_by_th`: đổi dấu class ID của box dưới ngưỡng positive.

Không chạy lại ensemble/detector, không train, không dùng DetZero/GRM/PRM. Thuật toán tracking/refinement dùng source MS3D++ hiện có. Wrapper chỉ bridge dữ liệu, chuẩn hóa biểu diễn tương đương và instrument số liệu/snapshot; không sửa source core.

## Tham số được chọn

Dùng nguyên các ngưỡng của `target_custom/label_generation/round1/cfgs/ps_config.yaml`, lưu snapshot tại output. Các ngưỡng theo số frame được giữ nguyên cho một baseline có thể tái lập.

| Tracking | Score tạo/cập nhật chính | Association | Ngưỡng cost | Max age update | Min hits birth |
|---|---:|---|---:|---:|---:|
| veh_all |0.7|GIoU3D|1.0|2frame|2|
| veh_static |0.5|IoU2D|0.5|3frame|2|
| pedestrian |0.3|GIoU3D|0.8|2frame|2|

Association dùng Hungarian/bipartite trên cost `1-IoU/GIoU`. KF có10state, đo7giátrị; F dùng dt từ timestamp thực. Các MAX_AGE/MIN_HITS vẫn đếm frame, không phải giây. Tracker native xuất mọi track còn trong danh sách (kể cả birth/unassociated); confidence/độ dài được lọc tiếp ở refinement.

| Refinement | Giá trị |
|---|---|
| POS_TH Vehicle/Pedestrian |0.7 /0.6|
| veh_all: detections score>0.7 tối thiểu |4|
| veh_static: detections score>0.7 tối thiểu |5|
| static vehicle track length tối thiểu |10box|
| pedestrian: detections score>0.6 tối thiểu |2|
| pedestrian track length tối thiểu |10box|
| Giữ track pedestrian đứng yên |false|
| Rolling KBF window |15quan sát, khoảng0.75s ở20Hz nếu liên tục|
| Static score floor khi đủ điều kiện |0.8|
| Propagate ngoài track |tối đa10frame mỗi phía (~0.5s), trong cùng sequence|
| Propagation score |score biên ×0.99^k, clip tối thiểu0.3|
| Final NMS Vehicle/Pedestrian |0.05 /0.5|
| Point support |num_pts>1, XY margin0.1m trong hàm native|
| Ego removal |abs(x)<1m AND abs(y)<1m|

Motion state Vehicle được coi tĩnh khi khoảng cách tâm đầu–cuối<1m và trung bình variance XY<0.1; đối chiếu với bộ track còn lại có thể đổi thành dynamic. Pedestrian dùng ngưỡng đầu–cuối2m và variance0.1. KBF/refinement offline có thể dùng quan sát tương lai ở đầu track; không phải thuật toán online nhân quả.

## Output đã tạo và semantics

Thư mục riêng `runs/ms3d_three_model_round1`:

```text
input_fusion3_original.pkl
initial_pseudo_labels.pkl
ps_config.yaml
run_metadata.json
tracks_world_veh_all.pkl
tracks_world_veh_static.pkl
tracks_world_ped.pkl
stages/                         # snapshot sau motion/merge/rolling/propagation
final_ps_dict.pkl               # định dạng native, gồm class âm
final_positive_only.pkl         # bản chỉ lấy class dương để xem/export
tracking_summary.json
refinement_audit.json
point_support_audit.json
nms_audit.json
summary.json
logs/pipeline.log
```

Mỗi hàng label: `[x,y,z,dx,dy,dz,yaw,class_id,score]`. `class_id=-1/-2` là box dưới POS_TH, không phải positive label. Loader train hiện tại lọc ID dương. `num_pts` và `memory_counter` giữ cùng số hàng với gt_boxes.

Final không bảo toàn mọi singleton score cao bằng mọi giá: native final NMS, point support và các ngưỡng positive phục vụ tạo nhãn tin cậy có thể loại/đánh ignore box. Số thay đổi ở từng khâu được báo bên dưới; tăng/giảm số box không phải thay đổi accuracy khi chưa có GT.

## Chạy lại / tiếp tục

```bash
docker exec ms3d-fusion-20261105 python tools/run_ms3d_from_three_model_fusion.py

# Nếu tracking đã lưu và chỉ cần refinement:
docker exec ms3d-fusion-20261105 python tools/run_ms3d_from_three_model_fusion.py --phase refine
```

Runner kiểm tra SHA256 nguồn/config trước khi dùng cache. Không tái sử dụng cache nếu input khác. `.get_lidar()` trả bản copy để các phép cộng tọa độ in-place của source không sửa cache point cloud.


## Kết quả đã thực thi

Nguồn input 8,414 box không bị sửa. Đầu ra native có 9,057 box trên 204 frame; trong đó **7,205 nhãn dương** và **1,852 ignore**.

| Tập nhãn | Vehicle | Pedestrian | Tổng |
|---|---:|---:|---:|
| Fusion đầu vào, mọi score đã lưu |8,057|357|8,414|
| Fusion đầu vào nếu áp POS_TH 0.7/0.6 trực tiếp |5,412|153|5,565|
| MS3D++ cuối: nhãn dương |6,881|324|7,205|
| MS3D++ cuối: ignore |1,806|46|1,852|
| MS3D++ cuối: toàn bộ native |8,687|370|9,057|

Không có Cyclist. So cùng ngưỡng positive, cuối tăng1,640 box so với input (Vehicle+1,469/Pedestrian+171). Một phần do tracker/refinement nâng score, một phần do tạo proposal theo thời gian; không coi tất cả là đối tượng mới hay cải thiện accuracy.

### Tracking và lọc track

| Nhánh | Track thô | Sau confidence | Sau length | Sau merge/refinement |
|---|---:|---:|---:|---:|
| veh_all |363|151|không lọc riêng length|126|
| veh_static |462|115|86|86|
| pedestrian |13|3|3|3|

Tracking chạy lần lượt khoảng61.38s,27.42s,1.59s. Các nhánh không cộng trực tiếp với nhau vì cùng một đối tượng có thể xuất hiện trong cả hai nhánh xe. Track quá ngắn/không đủ box confidence cao bị loại khỏi nhánh refinement; input fusion vẫn được đưa vào bước update_ps, nên xóa track không đồng nghĩa xóa hết detection tương ứng.

Sau đối chiếu motion/merge: veh_all123static+3dynamic; veh_static84static+2dynamic; pedestrian3dynamic. Đây là phân loại do heuristic của MS3D++, chưa có GT xác nhận trạng thái chuyển động.

### Rolling KBF và propagation

- Static tracks có5,848 quan sát. Rolling KBF tạo5,848 box làm mượt, dùng cửa sổ15 quan sát.
- Propagation tạo tổng6,984 track-frame proposal:5,848 vị trí trong track và1,136 vị trí mở rộng ngoài biên track. Không có frame nội bộ mới được chèn trong lần chạy này.
- Con số1,136 là proposal trước lọc/NMS, có thể trùng nhãn gốc hoặc nhánh khác. Nó không phải số nhãn cuối được thêm.
- Chỉ track được coi static được sử dụng trong nhánh static khi update_ps. Nhánh veh_all giữ các track đã qua lọc; score dưới0.7 được nâng lên0.7. Pedestrian còn lại có score floor0.6.
- Source native chỉ thêm pedestrian track ở frame vốn có pedestrian trong initial pseudo-label. Vì vậy đây không phải bước khôi phục người trên mọi frame trống detection.

### Ghép, NMS và point support

| Bước | Số proposal/box |
|---|---:|
| Ghép initial + veh_all + static/ped proposals |21,165|
| NMS loại proposal trùng/chồng |12,070|
| Sau NMS, đưa vào đếm điểm |9,095|
| Loại vì num_pts<=1 |38|
| Loại tâm nằm trong ego rectangle |0|
| Còn trước khi gán ignore |9,057|
| POS_TH đổi class sang âm |1,852|
| Nhãn dương cuối |7,205|

`nms_audit.json`/`summary.json` gốc ghi21,149 candidates và9,079 retained cho các lần native gọi NMS trên mảng>1box. Có16 trường hợp singleton đi qua nhánh bypass. File `candidate_accounting.json` bổ sung kế toán tổng chính xác:21,165 ->9,095. 12,070 proposal bị NMS loại phần lớn là các nhánh đề xuất chồng nhau; không diễn giải thành12,070 vật thể bị mất.

NMS Vehicle0.05 khá mạnh. Singleton score>0.7 vẫn có thể bị box track/static lấn, hoặc bị point support loại. Điểm được đếm trên cloud prepared của **frame hiện tại, một sweep**, không dùng cloud tích lũy làm điều kiện giữ nhãn.

## Kiểm tra độc lập và các frame cần xem lại

[verification.json](../../../runs/ms3d_three_model_round1/verification.json): **PASS**.

- 204 frame đúng ID/thứ tự; SHA256 fusion/manifest/config không đổi.
- Chuẩn hóa388 box bảo toàn footprint: BEV IoU nhỏ nhất0.9999826 (sai số số học), center/Z/score không đổi.
- Tất cả9,057 box hữu hạn, kích thước dương, num_pts/memory_counter khớp số hàng.
- Tính lại point support độc lập cho mọi box, khớp chính xác100%; mọi box còn lại có ít nhất2điểm theo margin native0.1m.
- Class dương/âm khớp POS_TH; positive-only chính xác là tập con class>0 của native final.
- Propagation nằm trong sequence và không quá10frame ngoài biên track; không có ego-center box còn sót.
- 5 test bridge/geometry/native-ignore PASS; 6 test viewer data-contract PASS.

[output_changes.json](../../../runs/ms3d_three_model_round1/output_changes.json) ghép box final dương với input cùng lớp bằng Hungarian BEV IoU>=0.1:6,500 box ghép được,705 không có correspondence theo tiêu chí này. Đây là correspondence với prediction đầu vào, không phải recall/precision theo GT.

Trong6,500 cặp: tâm dịch trung bình0.0588m, median0.0257m, P95=0.2128m, max1.8801m. Ưu tiên xem:

| Frame | Dịch tâm 3D | Ghi chú |
|---|---:|---|
| frame_00123 |1.880m|Chủ yếu XY (~1.880m), hướng box cũng thay đổi mạnh|
| frame_00115 |1.494m|Chủ yếu XY (~1.494m)|
| frame_00197 |1.452m|Chủ yếu Z (~1.383m)|
| frame_00201 |1.442m|Chủ yếu Z (~1.382m)|

Các cặp trên có thể chịu ảnh hưởng association/rolling KBF/propagation hoặc correspondence trước–sau chưa đúng đối tượng. Cần kiểm tra hình học bằng BEV/3D trước khi tin nhãn để train. Point support PASS chỉ chứng minh có điểm trong box; không chứng minh box đúng vật thể hoặc đúng kích thước.

Mẫu dữ liệu chỉ~10.14s,20Hz. Giữ số frame mặc định khiến max-age/window/propagation ngắn hơn theo giây so với tập10Hz; số track static nhiều cần xem thực tế. Chưa tune theo GT và chưa train mô hình trên pseudo-label mới.

## File dùng và cách xem

- [final_ps_dict.pkl](../../../runs/ms3d_three_model_round1/final_ps_dict.pkl): file pseudo-label native cuối, gồm class âm để biểu diễn ignore.
- [final_positive_only.pkl](../../../runs/ms3d_three_model_round1/final_positive_only.pkl): chỉ7,205 nhãn dương, dùng để xem/export.
- [summary.json](../../../runs/ms3d_three_model_round1/summary.json), [candidate_accounting.json](../../../runs/ms3d_three_model_round1/candidate_accounting.json): thống kê cuối.
- [refinement_audit.json](../../../runs/ms3d_three_model_round1/refinement_audit.json): số track trước/sau từng filter, motion, merge, KBF và propagation.
- `stages/`: snapshot từng khâu; không ghi đè output fusion gốc.
- [code_fingerprints.json](../../../runs/ms3d_three_model_round1/code_fingerprints.json): SHA256 source core đã dùng.
- [pipeline.log](../../../runs/ms3d_three_model_round1/logs/pipeline.log), [verification.log](../../../runs/ms3d_three_model_round1/logs/verification.log): bằng chứng chạy thật.

Viewer thêm case **MS3D++ · pseudo-label cuối** (`ms3d_final`), nạp positive-only; bộ lọc nguồn để Tất cả vì final không còn provenance từng detector. Bật overlay `fusion3` để so trước/sau ở BEV và3D.

```bash
cd /home/khanhhuy/MS3D
python3 tools/serve_local_viewer.py --port 8766
```

Mở `http://127.0.0.1:8766/?frame=123&case=ms3d_final&score=0.3`. Nếu đang chạy server cũ, cần khởi động lại để catalog nạp case mới, hoặc dùng cổng khác.

Chạy lại kiểm tra độc lập:

```bash
docker exec ms3d-fusion-20261105 python tools/test_ms3d_fusion_bridge.py
docker exec ms3d-fusion-20261105 python tools/verify_ms3d_pseudo_labels.py
python3 tools/test_local_viewer.py
```


## Tra source theo từng khâu

| Khâu | File / hàm | Input → output |
|---|---|---|
| Bridge | [ManifestDataset](../../../tools/run_ms3d_from_three_model_fusion.py#L50) | manifest + prepared XYZ → pose/timestamp/cloud theo frame |
| Tracking | [get_tracklets](../../../pcdet/utils/tracker_utils.py#L12) | label trong vehicle → track world theo từng nhánh |
| Lọc/motion/merge | [refine_veh_labels](../../../pcdet/utils/ms3d_utils.py#L115) | track thô → track đủ confidence và motion state |
| Rolling KBF | [get_track_rolling_kde_interpolation](../../../pcdet/utils/ms3d_utils.py#L371) | quan sát static nhiều frame → box tổng hợp theo cửa sổ |
| Propagation | [propagate_static_boxes](../../../pcdet/utils/ms3d_utils.py#L426) | static box/pose/sequence → proposal mở rộng có suy giảm score |
| Pedestrian | [refine_ped_labels](../../../pcdet/utils/ms3d_utils.py#L163) | track người → track đủ độ dài/confidence, lọc static |
| Ghép/lọc | [update_ps](../../../pcdet/utils/ms3d_utils.py#L23) | initial + world tracks → vehicle labels sau NMS/points/ego |
| Positive/ignore | [select_ps_by_th](../../../pcdet/utils/ms3d_utils.py#L188) | gt_boxes + POS_TH → class dương/âm |

## Kiểm tra viewer cuối

12 nhóm Chrome/WebGL QA PASS với8case, có chuyển sang `ms3d_final`, kiểm tra số box khớp API. Không lỗi JS/console hoặc request ngoài localhost. Bằng chứng riêng, giữ nguyên QA cũ:

- [browser_verification.json](../../../runs/ms3d_three_model_round1/viewer_qa/browser_verification.json).
- [final_case_review.json](../../../runs/ms3d_three_model_round1/viewer_qa/final_case_review.json): frame115/123/197/201, nhãn cuối + overlay fusion3.
- [Ảnh BEV/3D frame123](../../../runs/ms3d_three_model_round1/viewer_qa/frame_00123_final_vs_fusion3.png):37positive cuối so với46input, cùng score slider0.3. Hai bộ đã được lọc đầu vào khác nhau; slider thấp không khôi phục box ignore.

Server QA8766/Chrome tạm được dừng sau kiểm tra. Server đang có sẵn trên8765 được giữ nguyên; muốn nạp case mới cần restart server đó hoặc chạy lệnh8766 ở trên.
