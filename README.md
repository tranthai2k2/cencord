# NSFW Tool

Công cụ Tkinter phát hiện và làm mờ vùng nhạy cảm trong ảnh/GIF bằng các model YOLO.

---

## Cài đặt

```bash
pip install -r requirements.txt
```

Nếu bạn dùng Python từ WebUI Forge:
```bash
# Dùng Python của WebUI (ví dụ)
E:\webui_forge_cu121_torch231\python_embeded\python.exe -m pip install -r requirements.txt
```

---

## Cách dùng

### Chạy ứng dụng

```bash
python nsfw_tool.py
```

Ứng dụng có các tab Blur, Review và Train. Model YOLO đặt trong `model-yolo/`
được tự động quét; các file model lớn không được commit vào repository.

---

## Kiểm tra nhanh

```bash
python nsfw_tool.py --selftest
python mp4_to_gif.py --selftest
```

## Model

Ứng dụng dùng các model YOLO trong `model-yolo/`. Vì dung lượng lớn, model
`.pt` chỉ được giữ local và bị loại khỏi Git; cần chép model vào thư mục này
trước khi chạy tab Blur.

---

## Công cụ kèm theo

- `prepare_dataset.py`: chuẩn bị dataset LabelMe/YOLO cho quá trình train.
- `mp4_to_gif.py`: giao diện chuyển video thành GIF và hỗ trợ blur vùng chọn.
- `model-yolo/pyinstxtractor.py`: tiện ích trích xuất PyInstaller.

## Lưu ý

- Cần Python có Tkinter để chạy giao diện.
- Có thể dùng CPU; GPU CUDA được dùng khi PyTorch/Ultralytics nhận diện được.
- Không commit dữ liệu người dùng, model hoặc output vào repository.
