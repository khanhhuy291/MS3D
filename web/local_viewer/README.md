# MS3D Local Inspector

Web local đọc kết quả đã lưu, xem đồng thời BEV và 3D. Server dùng thư viện chuẩn Python và NumPy; frontend Three.js r160/OrbitControls được lưu trong `vendor/` kèm MIT license, không cần CDN hoặc npm khi chạy.

Từ repo:

```bash
python3 tools/serve_local_viewer.py --port 8765
```

Mở `http://127.0.0.1:8765`. Giữ terminal chạy server; `Ctrl+C` dừng. Đổi `--port` nếu cổng đang được dùng. Từ thư mục bất kỳ:

```bash
python3 /home/khanhhuy/MS3D/tools/serve_local_viewer.py --port 8765
```

Xem [hướng dẫn, nguồn dữ liệu và kiểm tra](../../docs/experiments/two_model_detection/LOCAL_VIEWER.md).
