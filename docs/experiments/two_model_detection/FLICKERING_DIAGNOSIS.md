# Chẩn đoán flickering sau MS3D++ — 2026-11-07

Đã đọc source và đo trên artifacts của `runs/ms3d_three_model_round1`. Chỉ chẩn đoán, không đổi ngưỡng hoặc ghi đè nhãn baseline.

## Kết luận có bằng chứng

Pseudo-label cuối được ghép từ detection ban đầu và track/refined proposals, rồi NMS/point support/positive threshold. Nó vẫn có detection không được track dài hạn hỗ trợ. Do đó có tracking không đảm bảo nhãn cuối liên tục cho mọi box.

### 1. Detection gốc còn ngoài track tốt, score vượt/rụng ngưỡng theo frame

Đo trên7,205positive cuối bằng same-class BEV IoU>=0.5 với proposals thuộc các track được giữ:

- 7,020positive có hỗ trợ hình học từ ít nhất một nhánh track/refinement.
- 185positive không có hỗ trợ theo tiêu chí trên.
- Ghép final giữa hai frame kế cận trong **world**, cùng lớp, Hungarian BEV IoU>=0.3: có104positive ở frame nội bộ không ghép được ở cả frame trước lẫn frame sau. Tất cả104 thuộc nhóm không có kept-track support>=0.5.
- 104 là số lần xuất hiện box theo frame, không phải104đối tượng. Association có thể hụt vì hình học đổi mạnh; không có GT để xác nhận danh tính.

Ví dụ vùng gần `x=-0.65,y=31.50,z=1.15` trong vehicle ở frame3. Chuyển tâm sang world và tìm ứng viên Vehicle trong1m XY ở từng frame:

| Frame | Score input | Kết quả final native | Viewer positive-only |
|---|---:|---|---|
|0|0.77949|class+1|hiện|
|1|0.80731|class+1|hiện|
|2|0.69351|class-1, ignore|mất|
|3|0.74002|class+1|hiện|
|4|0.65513|class-1, ignore|mất|
|5–7|không có ứng viên|không có box|mất|
|8|0.71932|class+1|hiện|
|9–10|không có ứng viên|không có box|mất|
|11|0.72994|class+1|hiện|

Score cuối bằng input trong ví dụ, nên temporal processing không khôi phục tính liên tục tại vùng này. Đây là ứng viên theo vị trí world, chưa khẳng định cùng vật thể theo GT. File đo: [flicker_roi_example.json](../../../runs/ms3d_three_model_round1/flicker_roi_example.json).

Source [update_ps](../../../pcdet/utils/ms3d_utils.py#L23) vẫn giữ `cur_veh_boxes` rồi ghép với track boxes. Việc track ngắn bị xóa trong refine không tự xóa detection ban đầu tương ứng. [select_ps_by_th](../../../pcdet/utils/ms3d_utils.py#L188) đánh ignore theo từng frame; viewer case `ms3d_final` chỉ nạp class dương.

### 2. Pedestrian bị mất ở bước xuất dù track vẫn có box

Ba pedestrian track đã lọc có326track-frame slots.324 có positive tương ứng;2slot mất đều ở `frame_00007`, track0 vàtrack1. Frame6 vàframe8 có nhãn, frame7 không có.

Nguyên nhân source rõ ràng: trong `update_ps`, chỉ ghép `track_boxes_ped` khi `cur_ped_boxes.shape[0] != 0`. Nếu frame không có detection người ban đầu thì trả mảng người rỗng, kể cả track đã có box cho frame đó. Đây là điều kiện trong code hiện tại; không phải tracker không dự đoán được.

### 3. Track xe đã giữ không có lỗ hình học bên trong khoảng được xuất

Ở cả hai ngưỡng BEV IoU>=0.3 và>=0.5:

- `veh_all`:5,534/5,534slot có final positive phủ tương ứng.
- `veh_static` thực sự dùng trong update (84trackstatic, bỏ2dynamic):6,893/6,893slot có final positive phủ tương ứng.
- Không có đoạn mất hình học nội bộ theo phép đo này.

Hai nhánh trùng đối tượng; các slot không cộng thành số nhãn unique. Best-IoU coverage có thể khớp box lân cận, không chứng minh giữ nguyên track ID hoặc hình học ổn định. Kết quả chỉ cho thấy chưa có bằng chứng NMS/point support làm thủng các nhánh xe đã giữ theo tiêu chí này. Toàn pipeline từng lọc38box vì thiếu điểm; không nên coi38box đó là nguyên nhân chính của mọi flicker.

## Các cơ chế làm tình trạng rõ hơn

- MAX_AGE của nhánh xe chỉ2/3frame ở20Hz, tương đương~0.10/0.15s; giới hạn redundancy cũng ngắn. Track có thể chết rồi tái sinh khi miss kéo dài. KF dùng dt thực, nhưng MAX_AGE vẫn đếm frame.
- Vehicle cần ít nhất4 hoặc5 quan sát score>0.7 để nhánh track qua lọc, static còn cần>=10box. Vật thể phát hiện chập chờn dễ không được giữ track, trong khi detection gốc vẫn có mặt trong final ở những frame đủscore.
- Rolling KBF và propagation bổ sung chủ yếu cho static vehicle đủ điều kiện. Propagation10frame chỉ~0.5s, không đảm bảo khôi phục mọi vật thể hoặc đoạn miss dài.
- Final NMS Vehicle0.05 ghép các nguồn proposal theo score từng frame. Nó có thể đổi box được chọn, làm tâm/yaw/size giật. Chưa định lượng winner switching trong lượt chẩn đoán này; không đồng nhất hình học giật với box mất/hiện.
- Viewer score slider cao hơn ngưỡng positive có thể lọc thêm box được score-floor0.7/0.6. Khi kiểm tra nên đặt score0.3, tất cả lớp/nguồn/nhóm, chọn đúng case `ms3d_final`. Giảm slider không khôi phục ignore đã bị loại khỏi positive-only.

## Hướng sửa phù hợp với kết quả đo

1. Sửa riêng pedestrian export để lấy track ở cả frame không có pedestrian detection; giữ nguyên point support/NMS và đo lại frame7.
2. Tạo nhánh output mới có temporal confirmation/hysteresis theo **track**, kiểm soát các detection gốc không được track hỗ trợ. Tránh chỉ hạ ngưỡng score cho mọi frame.
3. Điều chỉnh tuổi track theo giây ở20Hz; bổ sung association và gap-fill ngắn có giới hạn thời gian/chuyển động/point support. Tăng MAX_AGE đơn lẻ không đảm bảo track qua các filter confidence.
4. Dùng lựa chọn box theo track ổn định và xử lý yaw theo đối xứng thay vì cho các nhánh thắng NMS luân phiên khi cần xử lý jitter hình học.

Mỗi thay đổi cần output riêng, so continuity và số box mất/ghost trước–sau. Chưa có GT nên không suy luận accuracy từ mức giảm flicker. Lượt này chưa sửa core hoặc tạo nhãn experimental.

## Tái lập

```bash
docker exec ms3d-fusion-20261105 python tools/analyze_ms3d_flicker.py
```

Script chỉ đọc nhãn/track và ghi audit mới. [flicker_diagnosis.json](../../../runs/ms3d_three_model_round1/flicker_diagnosis.json) có các thống kê và ví dụ; `flicker_veh_all_slots.json`, `flicker_veh_static_slots.json`, `flicker_ped_slots.json` chứa từng slot theo IoU0.3. Phép đo được lặp với IoU0.5 để kiểm tra độ nhạy cho missing slots. Các tiêu chí là geometric coverage, không phải track lineage/GT.
