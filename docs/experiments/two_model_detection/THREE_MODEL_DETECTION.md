# Thử nghiệm ba detector: Lyft, nuScenes và BEVFusion LiDAR-only

Trạng thái **HOÀN THÀNH 2026-11-05**: BEVFusion 204/204 frame, fusion ba model, 204 ảnh và video. 9 test và verification độc lập PASS. Không có GT để đo accuracy.

## Kết quả cuối

Score đầu ra >=0.3; singleton >=0.7. Số liệu là box-frame cộng qua 204 frame, không phải số vật thể duy nhất.

| Output | Vehicle | Pedestrian | Cyclist | Tổng |
|---|---:|---:|---:|---:|
| Lyft PV-RCNN++ | 6,634 | 346 | 1 | 6,981 |
| nuScenes VoxelRCNN | 11,470 | 645 | 23 | 12,138 |
| BEVFusion LiDAR-only | 6,654 | 22 | 2 | 6,678 |
| KBF gốc hai model | 4,619 | 314 | 2 | 4,935 |
| Fusion hai model giữ singleton | 7,274 | 357 | 0 | 7,631 |
| Fusion ba model giữ singleton | **8,057** | **357** | **0** | **8,414** |

Thêm BEVFusion làm tổng tăng **783 box Vehicle**, Pedestrian không đổi. Tăng số box không chứng minh tăng accuracy. BEVFusion riêng có ít dự đoán pedestrian ở ngưỡng này; kết quả chưa cho thấy nó bổ sung tốt cho lớp đó.

## So sánh trong vùng giao

Chỉ đếm box có **center** nằm đồng thời trong range LiDAR native của BEVFusion và range vehicle của OpenPCDet. Đây là phép lọc theo center, không yêu cầu toàn bộ box nằm trong vùng giao.

| Output | Vehicle | Pedestrian | Cyclist | Tổng |
|---|---:|---:|---:|---:|
| Lyft | 6,283 | 337 | 1 | 6,621 |
| nuScenes OpenPCDet | 11,235 | 567 | 23 | 11,825 |
| BEVFusion | 6,632 | 22 | 2 | 6,656 |
| Fusion hai model | 7,124 | 342 | 0 | 7,466 |
| Fusion ba model | 7,907 | 342 | 0 | 8,249 |

Phần tăng 783 Vehicle vẫn nằm trong vùng giao; không phải do đơn thuần mở rộng range đếm. Không so trực tiếp các tổng full range như một thước đo accuracy vì range/sweeps/feature khác nhau.

## Nguồn đóng góp sau NMS

| Thành viên nhóm | Box cuối |
|---|---:|
| Lyft + nuScenes | 4,793 |
| nuScenes + BEVFusion | 1,080 |
| Lyft + BEVFusion | 4 |
| Cả ba model | 62 |
| Chỉ nuScenes | 2,168 |
| Chỉ Lyft | 166 |
| Chỉ BEVFusion | 141 |

Một nhóm có tối đa một prediction từ mỗi model. Ba nguồn không thường xuyên cùng khớp đủ ba gate; phần bổ sung chủ yếu là cặp nuScenes–BEVFusion và box đơn BEVFusion. Không diễn giải 62 nhóm ba model là số true positives.

Trước score/NMS: BEVFusion ghép vào 63 cặp Lyft–nuScenes và 1,796 nhóm một nguồn. Có 9,124 ứng viên đạt score, NMS loại 710, còn 8,414. Các số ứng viên có thể chứa box trùng; không xem 710 là số vật thể bị bỏ sót.

## Mức giữ lại dự đoán score cao

Đối chiếu predictions gốc score>=0.7, cùng supercategory và BEV IoU>=0.5 với output (many-to-one):

| Nguồn | Prediction score>=0.7 | Được fusion 2 model phủ | Được fusion 3 model phủ | Mới phủ | Không còn phủ |
|---|---:|---:|---:|---:|---:|
| Lyft | 2,890 | 2,890 | 2,890 | 0 | 0 |
| nuScenes | 5,045 | 5,045 | 5,044 | 0 | 1 |
| BEVFusion | 1,109 | 982 | 1,107 | 125 | 0 |

Hai prediction BEVFusion score cao vẫn không được box cuối phủ ở mức IoU0.5; một prediction nuScenes từng được phủ nay không còn được phủ. Vì vậy không khẳng định mọi prediction mạnh đều được bảo toàn tuyệt đối. Đây là coverage **so với dự đoán**, không phải recall theo GT.

## Phạm vi và nguồn

- Giữ predictions Lyft và nuScenes từ thử nghiệm 204 frame đã hoàn thành.
- Người dùng thêm `checkpoints/lidar-only-det.pth`; checkpoint có 306 tensor, chỉ state_dict, input convolution 5 feature.
- Source/config BEVFusion host đã được kiểm tra tại `/home/khanhhuy/bevfusion-adapter`, mount chỉ đọc trong container riêng `ms3d-bevfusion-20261105`. Source host không có đủ `.so`; smoke lần đầu thất bại vì thiếu `ball_query_ext`. Đã chuyển runner sang source **build sẵn trong image** tại `/workspace/bevfusion-adapter`, có CUDA extensions tương thích. Không sửa source host.
- Config: `configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml`.
- Output riêng tại `runs/three_model_detection`; không sửa hoặc ghi đè predictions/fusion cũ.

## Tọa độ và box

BEVFusion chạy một sweep PCD FM trong hệ LiDAR-top gốc, input `[x,y,z,intensity,time_lag=0]`. Không transform input, không cộng Z, intensity scale=1. Range model gốc `[-54,-54,-5,54,54,3]`.

Output BEVFusion dùng LiDARInstance3DBoxes legacy: Z đáy và quy ước yaw khác OpenPCDet. Chuyển đổi dựa trên **corners thật của thư viện**, không chép trực tiếp 7 cột tensor:

1. Lấy 8 corners của từng box.
2. Áp dụng toàn bộ `T_vehicle_lidar.FM` lên corners.
3. Lấy hướng XY của cạnh corner 0→4 làm yaw theo OpenPCDet.
4. Tạo box upright bao toàn bộ corners trong hệ trục của hướng này; center XYZ là tâm box, không phải Z đáy.

Do FM extrinsic có roll/pitch, cuboid sau biến đổi bị nghiêng. Box yaw-only không biểu diễn chính xác độ nghiêng; upright enclosure có thể tăng kích thước so với box native. Lưu cả box/corners native và corners vehicle để kiểm tra lại. Runner kiểm tra synthetic box từ thư viện thật và đối chiếu center với gravity_center sau extrinsic.

Class mapping: car/truck/bus/trailer/construction_vehicle→Vehicle, pedestrian→Pedestrian, motorcycle/bicycle→Cyclist. Barrier và traffic_cone giữ trong raw native output, loại khỏi fusion ba supercategory.

## Fusion ba nguồn dự kiến

Tiếp tục giữ singleton score cao như thử nghiệm trước. Khởi tạo cặp Lyft–nuScenes bằng cùng assignment/ngưỡng để tái lập baseline hai model; sau đó ghép BEVFusion vào các nhóm, yêu cầu mọi cặp thành viên đều đạt gate. Một nhóm tối đa một box/model. Không cộng score như thể các model là các phép đo độc lập.

Center/yaw của nguồn score cao nhất; dimensions và score trung bình có trọng số score. Với nhóm chứa BEVFusion, căn chỉnh biểu diễn trục ngang trước khi trung bình dimensions: nếu trục X của một thành viên gần trục Y của nguồn mạnh hơn, đổi DX/DY của thành viên đó. Hai cách `(dx,dy,yaw)` và `(dy,dx,yaw+pi/2)` có cùng footprint, nhưng không được trung bình trực tiếp DX/DY khác trục. Không đổi geometry input hoặc yaw/center của nguồn được chọn. Nhóm chỉ có Lyft/nuScenes giữ nguyên quy tắc cũ để baseline hai model tái lập chính xác.

Admission 0.1, output 0.3, singleton 0.7, NMS Vehicle=0.5, Pedestrian/Cyclist=0.3. Ghi source indices/provenance để kiểm tra nguồn đóng góp.

Score các model chưa calibration. BEVFusion dùng intensity và một sweep; nuScenes OpenPCDet dùng 10 sweeps, Lyft dùng XYZ một sweep. Range BEVFusion hẹp hơn nên sẽ báo cả tổng full range và vùng giao theo center để tránh nhầm chênh lệch range với chất lượng.

## Kiểm tra input quan trọng

Toàn bộ 204 PCD có intensity bằng 0 (đã đếm trực tiếp từng field, không tự tạo intensity). BEVFusion nhận intensity=0 và time_lag=0. Vì detector này có sử dụng intensity, kết quả single-sweep này có domain gap về feature/sweeps; không coi là hiệu năng tối đa của checkpoint. Hai detector trước không dùng intensity.

## Tiến độ đã xác nhận

- Smoke BEVFusion: đủ 306/306 tensor, synthetic library corner conversion PASS.
- Full BEVFusion: 204/204 frame, median forward sau frame đầu 85.54 ms, peak CUDA allocated 453.07 MiB, không OOM.
- 7 test mới cho quy ước box, tilt enclosure, matching ba nguồn, chống bridging và source uniqueness đều PASS.
- Implementation hai nguồn tái lập chính xác cả geometry/score và source indices của baseline đã lưu ở toàn bộ 204 frame.
- Kiểm tra lần render đầu phát hiện 1,133 thành viên ở các nhóm có BEVFusion cần căn chỉnh DX/DY theo trục tham chiếu. Ví dụ cùng xe có box OpenPCDet khoảng `(4.53,1.88,yaw=1.55)` và BEVFusion khoảng `(2.22,4.79,yaw=3.08)`; trung bình thẳng làm box bè ngang. Đã bổ sung căn chỉnh trục và hai test hồi quy. Bản trước sửa được giữ tại `runs/three_model_detection/axis_alignment_before`, không dùng làm kết quả cuối.
- 9 test geometry/fusion sau sửa đều PASS; đã render lại đủ 204 ảnh và kiểm tra độc lập công thức căn chỉnh cùng toàn bộ provenance. Kết quả cuối có 1,128 thành viên cần đổi trục khi trung bình dimensions.

## Kiểm tra độc lập và nhận xét

- 306/306 BEVFusion checkpoint tensors và synthetic geometry test của thư viện thật: PASS.
- Corners được transform đúng, gravity center đúng, toàn bộ 8 corners nằm trong upright enclosure. Sai số số học vượt biên tối đa ~`2.077e-6 m`, dưới tolerance `3e-5 m`.
- Frame coverage/order đủ204, box finite/positive dimensions, source prediction được dùng tối đa một lần, mọi cặp trong nhóm thỏa matching gate: PASS.
- Tính lại độc lập score/dimensions sau căn chỉnh trục, singleton threshold, đếm output và provenance, decode đủ204 PNG: PASS.
- SHA256 predictions/baselines/manifest gốc không đổi. Kết quả thử nghiệm hai model giữ nguyên.
- Video H.264 1980×1320, 204 frame, 20 FPS, 10.2s; ffprobe đã đọc lại đủ204 frame.

Đã xem frame0/100/203: model BEVFusion bổ sung một số box ở các cụm điểm mà fusion hai model thiếu; ví dụ frame100 tổng tăng từ40 lên44. Hướng và kích thước giữa model có thể khác nhau, nên có thể xuất hiện box trùng hoặc geometry chưa tốt dù qua gate. Không xác nhận các box bổ sung đều đúng khi không có GT. Đặc biệt intensity toàn0 và chỉ một sweep là giới hạn quan trọng của lần thử BEVFusion này.

BEVFusion median inference sau frame đầu 85.54ms, peak CUDA allocated453.07MiB. Timer của runner BEVFusion bao gồm copy points lên GPU; không so trực tiếp như benchmark với timer forward thuần của OpenPCDet. Không phải FPS end-to-end.

## Artifacts

- [Video sáu ô](../../../runs/three_model_detection/comparison_20fps.mp4)
- [Ảnh frame0](../../../runs/three_model_detection/comparison/frame_00000.png), [frame100](../../../runs/three_model_detection/comparison/frame_00100.png), [frame203](../../../runs/three_model_detection/comparison/frame_00203.png)
- [Summary và provenance theo frame](../../../runs/three_model_detection/summary.json)
- [Verification](../../../runs/three_model_detection/verification.json)
- [BEVFusion metadata](../../../runs/three_model_detection/bevfusion/metadata.json)
- [Audit intensity](../../../runs/three_model_detection/input_intensity_audit.json)
- [BEVFusion standardized predictions](../../../runs/three_model_detection/bevfusion/result.pkl), [native boxes/corners](../../../runs/three_model_detection/bevfusion/native_geometry.pkl), [resolved config](../../../runs/three_model_detection/bevfusion/resolved_config.py)
- [Fusion ba model](../../../runs/three_model_detection/fusion.pkl), gồm `gt_boxes (N,9)`, `source_indices (N,3)`, `provenance`.
- Code: [inference BEVFusion](../../../tools/infer_bevfusion_third_model.py), [box adapter](../../../tools/bevfusion_box_adapter.py), [fusion](../../../tools/fuse_three_models.py), [tests](../../../tools/test_three_model_fusion.py), [verification](../../../tools/verify_three_model_fusion.py).
- Logs: `runs/three_model_detection/logs/bevfusion_smoke*.log`, `bevfusion_full.log`, `fusion_tests_axis_aligned.log`, `fusion_axis_aligned.log`, `verification.log`. Log và bản output trước sửa trục vẫn được giữ để audit, không phải output cuối.

Sáu ô video: hàng trên Lyft / nuScenes / BEVFusion; hàng dưới KBF gốc / fusion hai model / fusion ba model. Nét đứt ở các ô fusion là singleton, nét liền là nhóm từ2/3nguồn. Màu cyan Vehicle, vàng Pedestrian, hồng Cyclist. Trục vehicle; ảnh hiển thị ±40m trong khi tổng đếm trên full range.

## Lệnh tái lập

Chạy từ `/home/khanhhuy/MS3D`. Hai container riêng được giữ lại; dùng `docker start` nếu đang stopped. Container BEVFusion dùng image `bevfusion:rtx5060`, mount MS3D tại `/workspace/MS3D`; runner dùng source build sẵn `/workspace/bevfusion-adapter`.

```bash
# Detector thứ ba; không chạy lại hai detector trước
docker exec ms3d-bevfusion-20261105 python /workspace/MS3D/tools/infer_bevfusion_third_model.py

# Test, fusion và verification trong container MS3D
docker exec ms3d-fusion-20261105 python tools/test_three_model_fusion.py -v
docker exec ms3d-fusion-20261105 python tools/fuse_three_models.py
docker exec ms3d-fusion-20261105 python tools/verify_three_model_fusion.py

# Tạo video; nếu file đã tồn tại, FFmpeg hỏi trước khi ghi đè
ffmpeg -hide_banner -loglevel warning -framerate 20 -start_number 0 \
  -i runs/three_model_detection/comparison/frame_%05d.png -frames:v 204 \
  -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart \
  runs/three_model_detection/comparison_20fps.mp4
```

Lần tạo container BEVFusion riêng đã dùng:

```bash
# Chỉ dùng nếu container chưa tồn tại
docker run -d --name ms3d-bevfusion-20261105 --gpus all --shm-size=2g \
  -v /home/khanhhuy/bevfusion-adapter:/workspace/bevfusion:ro \
  -v /home/khanhhuy/MS3D:/workspace/MS3D -w /workspace/bevfusion \
  --entrypoint /bin/bash bevfusion:rtx5060 -lc 'sleep infinity'
```

Mount `/workspace/bevfusion` chỉ phục vụ đọc source host. **Không đặt `--bev-repo` về mount host này**, vì nó thiếu CUDA `.so`; default của runner là source build sẵn bên trong image, nơi smoke/full run đã thành công.
