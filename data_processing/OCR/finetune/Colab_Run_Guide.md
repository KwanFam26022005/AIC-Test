# Hướng dẫn chạy Fine-tuning PP-OCRv6_medium_det trên Google Colab

Tài liệu này hướng dẫn bạn cách thiết lập môi trường, chuẩn bị dữ liệu và thực thi 3 phase fine-tuning model phát hiện văn bản tiếng Việt **PP-OCRv6_medium_det** trên Google Colab bằng cách nhân bản (git clone) kho chứa mã nguồn từ GitHub.

---

## Bước 1: Khởi tạo Notebook trên Google Colab
1. Truy cập [Google Colab](https://colab.research.google.com/).
2. Tạo một Notebook mới.
3. Kích hoạt GPU: Chọn **Runtime** -> **Change runtime type** -> Tại mục *Hardware accelerator* chọn **T4 GPU**, **L4 GPU** hoặc **A100 GPU** tùy vào tài khoản của bạn.

---

## Bước 2: Kết nối Google Drive & Cài đặt môi trường
Mở một cell mới trên Colab, copy đoạn code sau và chạy để mount Google Drive và cài đặt các thư viện cần thiết:

```python
# 1. Mount Google Drive để lưu checkpoint và logs
from google.colab import drive
drive.mount('/content/drive')

# 2. Cài đặt các thư viện bổ sung cần thiết
!pip install paddlepaddle-gpu -q
!pip install gdown albumentations -q

# 3. Clone Repository chứa code fine-tune của bạn
%cd /content
!git clone https://github.com/KwanFam26022005/AIC-Test.git
%cd /content/AIC-Test

# 4. Clone PaddleOCR chính thức để sử dụng engine huấn luyện
%cd /content
import os
if not os.path.exists('PaddleOCR'):
    !git clone https://github.com/PaddlePaddle/PaddleOCR.git
%cd /content/PaddleOCR
!pip install -r requirements.txt -q

print("\n[+] Thiết lập môi trường hoàn tất!")
```

---

## Bước 3: Tạo cấu trúc thư mục trên Google Drive
Chạy cell dưới đây để tạo cấu trúc thư mục lưu trữ thống nhất trên Google Drive:

```python
import os

drive_base = "/content/drive/MyDrive/OCR_finetune"
subdirs = ["pretrained", "data", "checkpoints", "configs", "logs", "scripts"]
for sd in subdirs:
    os.makedirs(os.path.join(drive_base, sd), exist_ok=True)

print(f"[+] Đã khởi tạo các thư mục lưu trữ tại: {drive_base}")
```

Sau khi chạy xong, hãy copy hoặc di chuyển toàn bộ nội dung của repository `AIC-Test` vào thư mục Drive tương ứng để phục vụ quá trình huấn luyện:
- Copy tất cả các file cấu hình `.yml` từ thư mục `configs/` vào `/content/drive/MyDrive/OCR_finetune/configs/`
- Copy tất cả các script `.py` từ thư mục `scripts/` vào `/content/drive/MyDrive/OCR_finetune/scripts/`

Bạn có thể thực hiện việc copy tự động bằng cell lệnh sau trên Colab:

```bash
# Copy configs
!cp -r /content/AIC-Test/data_processing/OCR/finetune/configs/* /content/drive/MyDrive/OCR_finetune/configs/

# Copy scripts
!cp -r /content/AIC-Test/data_processing/OCR/finetune/scripts/* /content/drive/MyDrive/OCR_finetune/scripts/

print("[+] Đã đồng bộ configs và scripts lên Google Drive!")
```

---

## Bước 4: Tải Pretrained Weights của PP-OCRv6_medium_det
Chạy cell này để tải pretrained weights từ HuggingFace lưu trữ trực tiếp lên Google Drive:

```python
import os

weights_path = "/content/drive/MyDrive/OCR_finetune/pretrained/PP-OCRv6_medium_det.pdparams"
if not os.path.exists(weights_path):
    print("[*] Đang tải pretrained weights PP-OCRv6_medium_det...")
    url = "https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det/resolve/main/student.pdparams"
    !wget -O {weights_path} {url}
    print("[+] Đã tải xong weights!")
else:
    print("[+] Đã có sẵn pretrained weights trên Google Drive.")
```

---

## Bước 5: Chuẩn bị bộ dữ liệu VinText
Chạy script `prepare_vintext.py` để tự động tải, giải nén và phân chia dữ liệu VinText (Train 1200 / Val 300 / Test 500) đồng thời chuyển đổi nhãn sang định dạng PaddleOCR:

```bash
# Chạy script chuẩn bị dữ liệu
!python /content/drive/MyDrive/OCR_finetune/scripts/prepare_vintext.py \
    --output_dir /content/drive/MyDrive/OCR_finetune/data/vintext \
    --download_dir /content/tmp_vintext
```

---

## Bước 6: Chạy Master Training Runner (Tự động 3 Phase)
Chạy cell dưới đây để bắt đầu huấn luyện. Script sẽ tự động nhận diện GPU của bạn, chọn cấu hình tối ưu, tự động resume từ checkpoint gần nhất trên Drive nếu bị ngắt kết nối đột ngột:

```python
# Di chuyển vào thư mục PaddleOCR để import các module hệ thống chính xác
%cd /content/PaddleOCR

# Thực thi huấn luyện tự động
!python /content/drive/MyDrive/OCR_finetune/scripts/train_runner.py --drive_dir /content/drive/MyDrive/OCR_finetune
```

*Lưu ý: Nếu Colab bị disconnect hoặc hết thời gian chạy, bạn chỉ cần mở lại notebook, chạy lại **Bước 2** để mount Drive & cài đặt thư viện, sau đó chạy lại cell của **Bước 6** này để tiếp tục tự động huấn luyện từ epoch đang dở.*

---

## Bước 7: Đánh giá mô hình trên tập Test
Sau khi hoàn thành huấn luyện, chạy cell dưới đây để đánh giá độ chính xác (Precision, Recall, Hmean) của mô hình trên tập dữ liệu Test độc lập (500 ảnh) của VinText:

```python
eval_script = "/content/drive/MyDrive/OCR_finetune/scripts/evaluate.py"
%cd /content/PaddleOCR

# Đánh giá Phase 1 (Freeze backbone)
print("=== ĐÁNH GIÁ PHASE 1 ===")
config1 = "/content/drive/MyDrive/OCR_finetune/configs/finetune_phase1.yml"
ckpt1 = "/content/drive/MyDrive/OCR_finetune/checkpoints/phase1_freeze/best_accuracy"
if os.path.exists(ckpt1 + ".pdparams"):
    !python {eval_script} --config {config1} --checkpoint {ckpt1} --phase 1 --epoch 40

# Đánh giá Phase 2 (Unfreeze all)
print("\n=== ĐÁNH GIÁ PHASE 2 ===")
config2 = "/content/drive/MyDrive/OCR_finetune/configs/finetune_phase2.yml"
ckpt2 = "/content/drive/MyDrive/OCR_finetune/checkpoints/phase2_unfreeze/best_accuracy"
if os.path.exists(ckpt2 + ".pdparams"):
    !python {eval_script} --config {config2} --checkpoint {ckpt2} --phase 2 --epoch 60

# Đánh giá Phase 3 (High-res)
print("\n=== ĐÁNH GIÁ PHASE 3 ===")
config3 = "/content/drive/MyDrive/OCR_finetune/configs/finetune_phase3.yml"
ckpt3 = "/content/drive/MyDrive/OCR_finetune/checkpoints/phase3_highres/best_accuracy"
if os.path.exists(ckpt3 + ".pdparams"):
    !python {eval_script} --config {config3} --checkpoint {ckpt3} --phase 3 --epoch 30
```

Kết quả chi tiết của từng lượt đánh giá sẽ được tự động ghi nhận vào file `/content/drive/MyDrive/OCR_finetune/logs/eval_history.csv` để bạn dễ dàng theo dõi và so sánh sự cải thiện qua các phase huấn luyện.
