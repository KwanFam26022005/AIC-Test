# Test on Google Colab — Object Detection Pipeline

Hướng dẫn step-by-step để test pipeline RAM++ → GroundingDINO trên Google Colab.

## 1. Upload test data

Upload thư mục `keyframe_test/L21_V001` (307 frames) lên Colab:

```python
# Option A: Upload từ Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy test data vào Colab (sửa path cho đúng)
!cp -r "/content/drive/MyDrive/AIC2026/keyframe_test/L21_V001" /content/keyframes_test/L21_V001
```

```python
# Option B: Upload trực tiếp qua Colab UI
# Click biểu tượng folder bên trái → Upload → chọn thư mục L21_V001
# Hoặc dùng zip:
!mkdir -p /content/keyframes_test/L21_V001
# (upload file .zip rồi unzip)
# !unzip L21_V001.zip -d /content/keyframes_test/L21_V001
```

## 2. Upload pipeline code

Upload 3 files vào `/content/pipeline/`:

```python
!mkdir -p /content/pipeline
# Upload các file này vào /content/pipeline/:
#   - ram_gdino_pipeline.py
#   - tag_canonicalization.py
#   - visualize.py
```

Hoặc clone từ repo:
```python
!git clone https://github.com/<your-repo>/AIC2026.git /content/AIC2026
# Symlink cho tiện
!ln -s /content/AIC2026/data_processing/object_detection /content/pipeline
```

## 3. Cài dependencies

```python
# ====== QUAN TRỌNG: Pin đúng transformers version ======
!pip install -q git+https://github.com/xinyu1205/recognize-anything.git
!pip install -q timm "transformers==4.46.3" accelerate pillow torch torchvision fairscale
!pip install -q tqdm supervision

# Verify version
import transformers
assert transformers.__version__ == "4.46.3", f"Wrong version: {transformers.__version__}"
print(f"✓ transformers=={transformers.__version__}")
```

## 4. Download RAM++ checkpoint

```python
!mkdir -p /content/pipeline/pretrained
!wget -q --show-progress -O /content/pipeline/pretrained/ram_plus_swin_large_14m.pth \
    https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth

import os
size_mb = os.path.getsize("/content/pipeline/pretrained/ram_plus_swin_large_14m.pth") / (1024**2)
print(f"✓ RAM++ checkpoint: {size_mb:.0f} MB")
```

## 5. Chạy pipeline

```python
%cd /content/pipeline

# Chạy trên 307 frames test
!python ram_gdino_pipeline.py \
    --input /content/keyframes_test/L21_V001 \
    --output /content/output/L21_V001.jsonl \
    --video-id L21_V001
```

**Thời gian ước tính (Colab T4):** ~10-15 phút cho 307 frames.

## 6. Validate output

```python
import json

lines = open("/content/output/L21_V001.jsonl").readlines()
print(f"Total frames: {len(lines)}")
print()

# In 5 frames đầu
for line in lines[:5]:
    doc = json.loads(line)
    print(f"  {doc['frame_id']}: "
          f"{len(doc['tags'])} tags, "
          f"{len(doc['objects'])} objects, "
          f"labels={doc['object_summary']}")

# Thống kê
all_objects = sum(len(json.loads(l)["objects"]) for l in lines)
all_tags = sum(len(json.loads(l)["tags"]) for l in lines)
print(f"\nTotal: {all_objects} objects, {all_tags} tags across {len(lines)} frames")
print(f"Avg: {all_objects/len(lines):.1f} objects/frame, {all_tags/len(lines):.1f} tags/frame")
```

## 7. Visualize kết quả

```python
!python visualize.py \
    --images /content/keyframes_test/L21_V001 \
    --metadata /content/output/L21_V001.jsonl \
    --output /content/output/visualized \
    --limit 20
```

Xem ảnh:
```python
from IPython.display import display
from PIL import Image
import glob

viz_images = sorted(glob.glob("/content/output/visualized/*.jpg"))[:10]
for path in viz_images:
    print(path)
    display(Image.open(path).resize((640, 360)))
```

## 8. Spot-check frame 089 (regression test)

Frame 089 từng có **147 objects** với pipeline cũ (do hierarchy collision).
Sau NMS dedup, expected ~20-30 objects.

```python
import json

with open("/content/output/L21_V001.jsonl") as f:
    for line in f:
        doc = json.loads(line)
        if doc["frame_id"] == "L21_V001_089":
            print(f"Frame 089:")
            print(f"  Tags ({len(doc['tags'])}): {doc['tags'][:10]}...")
            print(f"  Objects ({len(doc['objects'])}): {doc['object_summary']}")
            print(f"  Counts: {doc['object_counts']}")
            
            # REGRESSION: phải < 50 objects (147 cũ = bug)
            assert len(doc['objects']) < 50, f"Too many objects: {len(doc['objects'])}"
            print(f"  ✓ NMS dedup working ({len(doc['objects'])} < 50)")
            break
```

## 9. Giả lập tìm kiếm (Search Simulator)

Sau khi tạo xong file `.jsonl` chứa metadata của video, bạn có thể chạy thử công cụ tìm kiếm giả lập:

```python
# Chạy câu query trực tiếp từ CLI
!python search_simulator.py \
    --metadata /content/output/L21_V001.jsonl \
    --query "2 person and fish"
```

Hoặc chạy chế độ tương tác (interactive) để thử nhiều câu query khác nhau trực tiếp trên notebook:
```python
import sys
# Chạy interactive shell giả lập
!python search_simulator.py --metadata /content/output/L21_V001.jsonl
```

## 10. Download output

```python
from google.colab import files

# Download JSONL
files.download("/content/output/L21_V001.jsonl")

# Download visualized images (zip)
!cd /content/output && zip -r visualized.zip visualized/
files.download("/content/output/visualized.zip")
```

---

## Troubleshooting

### `ImportError: cannot import name 'apply_chunking_to_forward'`
→ transformers version sai. Phải dùng `transformers==4.46.3`. 
Restart runtime rồi chạy lại từ bước 3.

### `FileNotFoundError: RAM++ checkpoint not found`
→ Chưa download checkpoint. Chạy lại bước 4.

### `CUDA out of memory`
→ Thử dùng `grounding-dino-tiny` thay vì `base`:
```python
# Sửa trong ram_gdino_pipeline.py hoặc set env var:
import os
os.environ["GDINO_MODEL_ID"] = "IDEA-Research/grounding-dino-tiny"
```

### Resume nếu bị ngắt giữa chừng
```python
!python ram_gdino_pipeline.py \
    --input /content/keyframes_test/L21_V001 \
    --output /content/output/L21_V001.jsonl \
    --video-id L21_V001 \
    --resume
```
