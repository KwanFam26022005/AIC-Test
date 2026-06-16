# Video Data Processing Pipeline (AIC 2026)

Tài liệu này hướng dẫn chi tiết về cấu trúc thư mục, ý nghĩa các thành phần, và cách vận hành các bước xử lý dữ liệu video (trích xuất phân cảnh, trích xuất keyframes, và trích xuất đặc trưng AI).

---

## 📂 Cấu trúc thư mục

```text
data_processing/
├── keyframe_extraction/
│   ├── keyframe_extraction.py       # Script trích xuất ảnh keyframe từ kết quả của TransNetV2
│   └── transnetv2/                  # Thư mục chứa mô hình phân cảnh TransNetV2 (PyTorch)
│       ├── transnetv2_pytorch/      # Mã nguồn TransNetV2 PyTorch (model, inference)
│       └── setup.py                 # File cấu hình cài đặt thư viện cục bộ
└── feature_extraction/
    └── openclip_embedding.py        # Script trích xuất vector đặc trưng bằng OpenCLIP
```

### Ý nghĩa của từng mục:
*   **`keyframe_extraction/`**: Chứa toàn bộ mã nguồn xử lý liên quan đến việc phân rã video thành các shot cảnh và trích xuất ra các ảnh Keyframes đại diện.
    *   **`transnetv2/`**: Sử dụng mô hình Deep Learning **TransNetV2** để tìm ra ranh giới các cảnh (scenes/shots) của video.
    *   **`keyframe_extraction.py`**: Nhận danh sách phân cảnh từ TransNetV2 để tiến hành cắt frame ở giữa (mid-frame) của mỗi shot cảnh và lưu dưới dạng ảnh JPG kèm file ánh xạ CSV.
*   **`feature_extraction/`**: Chứa các script chạy các mô hình AI để chuyển ảnh keyframes thành các vector đặc trưng (Embeddings) phục vụ cho việc tìm kiếm ảnh/video.

---

## 🚀 Hướng dẫn chạy mô hình TransNetV2

Mô hình TransNetV2 được sử dụng để phân tích các shot chuyển cảnh trong video và xuất ra file kết quả danh sách các cảnh dạng `.scenes.txt` và xác suất dạng `.predictions.txt`.

### Bước 1: Khởi tạo và kích hoạt môi trường ảo
Môi trường ảo của dự án được đặt tại `data_processing/.venv` (Python 3.10). Kích hoạt bằng lệnh:
```bash
# Di chuyển đến thư mục transnetv2
cd keyframe_extraction/transnetv2

# Kích hoạt môi trường ảo
source ../../venv_transnetv2/bin/activate
```

### Bước 2: Cài đặt thư viện phụ thuộc (Editable Mode)
Đăng ký package cục bộ và cài đặt các thư viện cần thiết như `ffmpeg-python` (xử lý video) và `torch` (tương thích CUDA 12.4):
```bash
# Cài đặt gói transnetv2 ở chế độ editable
uv pip install -e .

# Gỡ bỏ bản ffmpeg cũ (nếu có) và cài đặt ffmpeg-python đúng chuẩn
uv pip uninstall ffmpeg
uv pip install ffmpeg-python

# Cài đặt PyTorch tương thích CUDA 12.4
uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### Bước 3: Chạy lệnh dự đoán (Inference)

Tùy vào nhu cầu, bạn có thể chạy dự đoán theo các chế độ dưới đây:

#### 1. Chạy cho 1 Video đơn lẻ
Đường dẫn đầu ra `-o` kết thúc bằng dấu `/` sẽ tự động tạo thư mục và sinh các file kết quả theo tên video:
```bash
CUDA_VISIBLE_DEVICES=7 ../../venv_transnetv2/bin/transnetv2_pytorch /tmp2/maitanha/vgu/ttn/data/AIC2025/videos/Videos_L21/L21_V001.mp4 -o /tmp2/maitanha/vgu/ttn/data/AIC2025/transnetv2/ --visualize
```

#### 2. Chạy cho 1 thư mục chứa các video (Không quét đệ quy)
Chương trình sẽ tự động lấy toàn bộ file video nằm trực tiếp trong thư mục đó:
```bash
CUDA_VISIBLE_DEVICES=7 ../../.venv/bin/transnetv2_pytorch /tmp2/maitanha/vgu/ttn/data/AIC2025/videos/Videos_L21/ -o /tmp2/maitanha/vgu/ttn/data/AIC2025/transnetv2/ --visualize
```

#### 3. Chạy quét toàn bộ các thư mục con (`--all_videos`)
Quét đệ quy qua toàn bộ cây thư mục `videos/*/*.mp4` để xử lý hàng loạt:
```bash
CUDA_VISIBLE_DEVICES=7 ../../.venv/bin/transnetv2_pytorch /tmp2/maitanha/vgu/ttn/data/AIC2025/videos -o /tmp2/maitanha/vgu/ttn/data/AIC2025/transnetv2/ --all_videos
```

---

## 📸 Hướng dẫn trích xuất Keyframes

Sau khi chạy xong TransNetV2, bạn tiến hành chạy file trích xuất keyframe để cắt ảnh và sinh file ánh xạ CSV.

```bash
# Đảm bảo đang đứng ở thư mục chứa file keyframe_extraction.py
cd /tmp2/maitanha/vgu/ttn/AIC2026/data_processing/keyframe_extraction

# Thực thi trích xuất bằng OpenCV tối ưu (Single-pass)
CUDA_VISIBLE_DEVICES=7 python keyframe_extraction.py
```
*   **Ảnh trích xuất**: Được lưu tại `/tmp2/maitanha/vgu/ttn/data/AIC2025/keyframes_transnetv2/Keyframes_LXX/LXX_VXXX/*.jpg`.
*   **File ánh xạ**: Được lưu tại `/tmp2/maitanha/vgu/ttn/data/AIC2025/map_keyframes/LXX_VXXX.csv`.

---

## 🧬 Hướng dẫn chạy Embedding Keyframe (Tính năng tương lai)

Hạng mục này phục vụ cho việc nạp ảnh keyframe đã trích xuất qua các mô hình Vision-Language để chuyển đổi thành vector đặc trưng (Embeddings) hỗ trợ tìm kiếm ngữ nghĩa.

*(Lệnh chạy mẫu cho mô hình OpenCLIP)*
```bash
# Di chuyển đến thư mục feature_extraction
cd /tmp2/maitanha/vgu/ttn/AIC2026/data_processing/feature_extraction

# Chạy trích xuất đặc trưng OpenCLIP
python openclip_embedding.py --input /tmp2/maitanha/vgu/ttn/data/AIC2025/keyframes_transnetv2 --output /tmp2/maitanha/vgu/ttn/data/AIC2025/embeddings/openclip
```

---

## 📝 Lộ trình công việc (Todo List)

### 1. Tiền xử lý dữ liệu (Preprocessing - TransNetV2)
- [x] Triển khai chạy suy diễn mô hình TransNetV2 tìm phân cảnh.
- [x] Triển khai trích xuất Keyframe từ kết quả phân cảnh của TransNetV2.
- [x] Tối ưu hóa thuật toán cắt keyframe bằng OpenCV chế độ 1 lượt đọc (Single-pass) để giảm thời gian xử lý.
- [ ] Viết test file để kiểm tra số lượng videos tải về, số lượng keyframes extract được so với kết quả của TransnetV2, ...


### 2. Trích xuất đặc trưng ảnh (Feature Extraction)
- [/] Implement OpenCLIP (SigLIP 2, ViT-H/14, CLIP-ViT-L/14,...)
- [ ] Implement BEiT-3 (Multimodal Foundation Model)
- [ ] Implement InternVL2 (Vision-Language Model)
- [ ] Implement DINOv3 / DINOv2 (Self-supervised Vision Features)

### 3. Tìm kiếm và các tính năng bổ trợ (Indexing & Downstream Tasks)
- [ ] Xây dựng cơ sở dữ liệu vector chỉ mục **FAISS (Faiss Index Build)**.
- [ ] Trích xuất văn bản trong ảnh bằng OCR.
- [ ] Nhận diện vật thể (Object Detection).
- [ ] Tích hợp tính năng Metadata-filtering và Text-search.
