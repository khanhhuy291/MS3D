# Commands — FM two-model detection

Chạy từ `/home/khanhhuy/MS3D` trên host. Container riêng hiện tại: `ms3d-fusion-20261105`, repo mount tại `/workspace/MS3D`.

## Môi trường đã dùng

```bash
docker run -d --name ms3d-fusion-20261105 --gpus all --shm-size=2g \
  -v /home/khanhhuy/MS3D:/workspace/MS3D -w /workspace/MS3D \
  --entrypoint /bin/bash bevfusion:rtx5060 -lc 'sleep infinity'

docker exec ms3d-fusion-20261105 python -m pip install \
  spconv-cu126==2.3.8 open3d==0.19.0 SharedArray scikit-image numba tensorboardX ninja pybind11

docker exec ms3d-fusion-20261105 python -m pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

docker exec ms3d-fusion-20261105 git config --global --add safe.directory /workspace/MS3D
docker exec ms3d-fusion-20261105 python setup.py build_ext --inplace
```

Image gốc đã có PyTorch 2.7.1+cu128, CUDA toolkit 12.8, `TORCH_CUDA_ARCH_LIST=12.0`, `MAX_JOBS=2`. spconv-cu126 đã chạy phép thử GPU trên sm_120 với runtime/toolkit của container này; tên wheel không thay thế cho kiểm tra thực tế.

## Chuẩn bị dữ liệu và test

```bash
docker exec ms3d-fusion-20261105 python tools/two_model_detection.py prepare
python3 tools/test_two_model_detection.py -v
```

`prepare` tạo cache trong `runs/two_model_detection/prepared`, manifest và config riêng. Timestamp đọc từ header; FM extrinsic được áp dụng đúng một lần; SHIFT_COOR=0.

## Smoke test (chỉ chạy sau khi build xong)

```bash
docker exec ms3d-fusion-20261105 python tools/two_model_detection.py infer \
  --model lyft --start 20 --limit 1
docker exec ms3d-fusion-20261105 python tools/two_model_detection.py infer \
  --model nuscenes --sweeps 10 --start 20 --limit 1
docker exec ms3d-fusion-20261105 python tools/two_model_detection.py compare
```

## Full sequence

```bash
docker exec ms3d-fusion-20261105 python tools/two_model_detection.py infer --model lyft
docker exec ms3d-fusion-20261105 python tools/two_model_detection.py infer --model nuscenes --sweeps 10
docker exec ms3d-fusion-20261105 python tools/two_model_detection.py compare --threshold 0.3
python3 tools/two_model_detection.py verify
```

Mỗi lần `infer` ghi lại result của model đang chọn trong thư mục output. Lưu smoke artifacts riêng trước khi full run. Đầu sequence chỉ dùng sweeps đã có, không lặp frame đầu để đủ số lượng. 10 sweeps ở 20 Hz tương đương khoảng 0.45 s lịch sử.

Nếu container đang dừng, dùng `docker start ms3d-fusion-20261105` trước các lệnh `docker exec`. Chỉ chạy `prepare` lại khi cần tạo lại cache/config; nó đọc raw và ghi lại artifacts trong output.

## Video sau khi đủ 204 ảnh

```bash
ffmpeg -hide_banner -loglevel warning -framerate 20 -start_number 0 \
  -i runs/two_model_detection/comparison/frame_%05d.png -frames:v 204 \
  -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart \
  runs/two_model_detection/comparison_20fps.mp4

ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames,duration \
  -of json runs/two_model_detection/comparison_20fps.mp4
```

Nếu video đã tồn tại, FFmpeg hỏi trước khi ghi đè. Lần chạy đã hoàn thành có đủ 204 ảnh, 204 frame video, summary và verification PASS.

## Output

```text
runs/two_model_detection/
  manifest.json
  geometry_checks.json
  configs/{lyft,nuscenes}.yaml
  prepared/frame_*.npy
  lyft/{metadata.json,result.pkl}
  nuscenes/{metadata.json,result.pkl}
  fusion/kbf.pkl
  comparison/frame_*.png
  summary.json
  logs/
```

Box ở vehicle frame: `[x,y,z,dx,dy,dz,yaw]`; file KBF thêm `class_id,score` thành 9 cột. Class supercategory: 1 Vehicle, 2 Pedestrian, 3 Cyclist. Ảnh mặc định hiển thị ±40 m; thống kê đếm trên toàn vùng model ±75.2 m.

Các commands trên là runbook. Xem PROGRESS.md và REPORT.md để phân biệt bước đã chạy với bước chưa chạy.
