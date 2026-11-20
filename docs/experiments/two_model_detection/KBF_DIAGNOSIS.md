# Vì sao KBF làm mất nhiều box — 2026-11-05

## Kết luận từ phép đo

Nguyên nhân chính là yêu cầu `DISCARD=2` kết hợp lọc score >=0.3 trước fusion. Box chỉ có một nguồn hỗ trợ có thể bị bỏ dù score rất cao. NMS không phải nguyên nhân chính trong run này.

Audit dùng lại dự đoán của 204 frame, không chạy detector lại và không ghi đè kết quả gốc. Baseline tái dựng từng bước khớp kết quả KBF đã lưu trên đủ 204 frame (kiểm tra toàn bộ giá trị box sau sắp xếp, tolerance 1e-5).

## Theo dõi qua các khâu

| Khâu | Số lượng trên 204 frame |
|---|---:|
| Tổng box hai model sau lọc score 0.3 | 19,119 |
| Box đầu vào có tham gia ít nhất một nhóm hợp lệ | 10,497 |
| Box đầu vào không vào nhóm hợp lệ | **8,622** |
| Box fusion ứng viên trước NMS | 4,994 |
| Ứng viên bị NMS loại | **59** |
| Box KBF cuối | 4,935 |

19,119 là tổng dự đoán từ hai nguồn, có các box cùng một đối tượng. Việc 10,497 box được hợp nhất thành 4,994 ứng viên có cả thao tác khử trùng bình thường; không được diễn giải chênh lệch này là số đối tượng bị bỏ sót. Tương tự 59 ứng viên bị NMS loại có thể bao gồm ứng viên trùng do các nhóm lân cận chồng nhau.

## Vì sao điều kiện nhóm loại box tốt theo score?

`get_matching_boxes()` tìm hàng xóm theo khoảng cách tâm 3D và yêu cầu ít nhất 2 box trong radius: Vehicle 1.5 m, Pedestrian 0.3 m, Cyclist 0.2 m. Tính cả chính box đang xét. `label_fusion()` còn yêu cầu ít nhất hai tâm khác nhau sau round 4 chữ số. Không có nhánh giữ lại singleton vì score cao.

| Chỉ tiêu | Lyft | nuScenes |
|---|---:|---:|
| Box score >=0.3 đầu vào | 6,981 | 12,138 |
| Không được dùng trong nhóm KBF hợp lệ | 2,795 | 5,827 |
| Trong số trên, score >=0.7 | 255 | 2,338 |
| Trong số trên, score >=0.9 | 4 | **864** |
| Box đơn lẻ có box từ model còn lại score <=0.3 trong radius | 692 | 174 |
| Box đơn lẻ vẫn có BEV IoU >=0.5 với model còn lại, nhưng tâm ngoài radius | 47 | 44 |

Các chỉ tiêu cuối có thể chồng nhau. “Không được dùng trong nhóm” không đồng nghĩa chắc chắn không có box đầu ra nào phủ lên vị trí đó. Đây là phép đo cơ chế fusion, không có ground truth để xác nhận đúng/sai.

Ví dụ thực tế ở `frame_00100`, hệ vehicle:

- nuScenes dự đoán box Vehicle tại `(3.566, 6.137, 0.786) m`, score **0.96046**.
- Box Lyft Vehicle score >=0.3 gần nhất cách **10.414 m**.
- Box này không vào nhóm hợp lệ, cũng không có partner Lyft score thấp trong radius.
- Tăng NMS không khôi phục được box chưa từng trở thành ứng viên fusion.

## Ảnh hưởng của lọc score trước fusion

Runner ban đầu dùng chung `--threshold=0.3` cho lọc nguồn trước fusion và hiển thị. Do đó các box score 0.1–0.3 được detector xuất ra đã bị loại trước khi có thể hỗ trợ một box score cao từ nguồn còn lại.

Đây là lựa chọn thử nghiệm khá bảo thủ. Với mục tiêu detection giữ độ bao phủ của hai model, nên tách ngưỡng admission và ngưỡng đầu ra, thay vì mặc định coi chúng giống nhau.

## Ablation chỉ thay fusion

Mọi dòng đều đếm đầu ra score >=0.3. Không thay dữ liệu, checkpoint, sweeps hoặc kết quả hai detector. Không render lại video và chưa chọn cấu hình mới làm kết quả chính.

| Cấu hình | Vehicle | Pedestrian | Cyclist | Tổng |
|---|---:|---:|---:|---:|
| Baseline admission 0.3, NMS xe 0.1 | 4,619 | 314 | 2 | 4,935 |
| Chỉ tăng NMS xe lên 0.5 | 4,637 | 314 | 2 | 4,953 |
| Admission 0.1, NMS như baseline | 6,244 | 406 | 6 | 6,656 |
| Admission 0.1 và NMS xe 0.5 | 6,378 | 406 | 6 | 6,790 |

Tăng NMS đơn lẻ chỉ thêm 18 box. Hạ admission thêm 1,721 box đầu ra, cho thấy lọc nguồn trước fusion có ảnh hưởng đáng kể. Thêm box không chứng minh tăng recall/precision khi chưa có GT. Hạ admission vẫn không cứu được box hoàn toàn không có partner.

## Vấn đề KDE với hai nguồn

4,408/4,994 nhóm baseline có đúng hai box; còn lại: 554 nhóm ba box, 29 nhóm bốn box, 3 nhóm năm box.

Source `kbf()` đang kiểm tra `boxes.shape == 0/1/2` thay vì số hàng `boxes.shape[0]`. Vì `shape` là tuple, nhánh special-case cho hai box không chạy; nhóm hai box đi vào KDE tổng quát, có thể dùng fallback khi covariance suy biến. Đây là lỗi source được xác nhận bằng đọc code, **chưa có ablation sửa lỗi để đo ảnh hưởng lên box hoặc accuracy**. Không quy 8,622 box bị loại do grouping cho lỗi này: chúng đã bị loại trước khi gọi KDE.

## Hướng thay đổi phù hợp với mục tiêu detection

1. Tách admission 0.1 và ngưỡng đầu ra 0.3; xem trực quan biến thể này trước khi chọn.
2. Với box không có partner, cân nhắc giữ riêng box score cao của từng model, rồi NMS theo lớp. Phải gọi rõ là biến thể **KBF + giữ singleton**, vì nó thay đổi cơ chế lọc của MS3D gốc.
3. Cân nhắc matching IoU cùng khoảng cách tâm để tránh loại các box còn chồng tốt nhưng tâm cách quá radius. Không tăng radius bừa vì có thể gộp hai xe đỗ sát nhau.
4. Sửa/test nhánh hai box riêng nếu tiếp tục dùng KBF với hai nguồn, lưu output thành thử nghiệm mới.

Trong MS3D còn có tracking/refinement sau KBF để khai thác thời gian và bổ sung một số nhãn. Lần thử hiện tại dừng ở detection/fusion nên không có các bước đó để bù một phần khoảng thiếu.

## Code và artifacts

- [Script chẩn đoán](../../../tools/analyze_kbf_detection.py)
- [JSON số liệu](../../../runs/two_model_detection/kbf_diagnosis/diagnosis.json)
- [Log chạy](../../../runs/two_model_detection/logs/kbf_diagnosis.log)
- `pcdet/utils/box_fusion_utils.py`: `get_matching_boxes`, `label_fusion`, `kbf`.
- `tools/two_model_detection.py`: cấu hình `discard`, `radius`, `nms`, `neg_th` và lọc nguồn.

Lệnh tái lập:

```bash
docker exec ms3d-fusion-20261105 python tools/analyze_kbf_detection.py
```

Không sửa code lõi KBF, cấu hình chính hoặc kết quả gốc trong lần chẩn đoán này.
