# Hướng dẫn Test Pipeline trên Google Colab

## 📋 Tổng quan

Pipeline gồm 4 stages chạy tuần tự trên Colab (GPU T4/A100):

```
Stage 1 (RAM++)  →  Stage 2 (GDINO)  →  Stage 3 (Co-DETR)  →  Stage 4 (Merge)
   env chính          env chính          env conda codetr        env bất kỳ
```

**Dữ liệu test:** 307 keyframes từ `keyframe_test/L21_V001/` (đã lưu trên Drive).

> ⚠️ **QUAN TRỌNG:** Stage 1+2 và Stage 3 chạy trên **2 môi trường khác nhau** vì xung đột thư viện:
> - Stage 1+2: `transformers==4.46.3` + torch mới
> - Stage 3: `torch==1.12.1` + `mmcv-full==1.6.0` + `mmdet==2.25.3`

---

## 🔧 Cell 0: Setup chung (Chạy 1 lần)

```python
# 0.1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 0.2. Clone repo
!git clone https://github.com/youngazier/AIC2026.git /content/AIC2026
%cd /content/AIC2026

# 0.3. Checkout branch Khoa (hoặc branch bạn đang dùng)
!git checkout Khoa

# 0.4. Tạo symlink tới keyframes trên Drive (thay đổi path cho đúng)
# Giả sử keyframes nằm ở: /content/drive/MyDrive/keyframe_test/L21_V001/
!ln -s /content/drive/MyDrive/keyframe_test/L21_V001 /content/keyframes_test

# 0.5. Kiểm tra
!ls /content/keyframes_test/ | head -5
!ls /content/keyframes_test/ | wc -l
# Kỳ vọng: 307 files
```

---

## 🏷️ Cell 1: Stage 1 + 2 — RAM++ & GroundingDINO

### Cell 1.1: Cài đặt môi trường (env transformers==4.46.3)

```python
# ====== Cài đặt RAM++ + GroundingDINO ======
# QUAN TRỌNG: pin transformers==4.46.3 — KHÔNG nâng cấp!
!pip install -q git+https://github.com/xinyu1205/recognize-anything.git
!pip install -q timm "transformers==4.46.3" accelerate pillow torch torchvision fairscale
!pip install -q supervision

# Kiểm tra version
import transformers
assert transformers.__version__ == "4.46.3", f"SAI VERSION: {transformers.__version__}"
print(f"✅ transformers=={transformers.__version__}")

# Tải checkpoint RAM++
!mkdir -p pretrained
!wget -q -O pretrained/ram_plus_swin_large_14m.pth \
    https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth
print("✅ Đã tải checkpoint RAM++")
```

### Cell 1.2: Chạy Stage 1 — RAM++ Tag Detection

```python
%cd /content/AIC2026/data_processing/object_detection

import sys, importlib
# Reload modules nếu chỉnh sửa
for mod_name in ['config', 'utils', 'stage1_ram_tagger']:
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])

from config import ensure_output_dirs, STAGE1_OUTPUT
ensure_output_dirs()

# Chạy Stage 1
from stage1_ram_tagger import run_stage1
from utils import Timer

with Timer("Stage 1 — RAM++"):
    stage1_results = run_stage1(input_dir="/content/keyframes_test")

# Kiểm tra output
print(f"\n📁 Output: {STAGE1_OUTPUT}")
print(f"   Exists: {STAGE1_OUTPUT.exists()}")
```

### Cell 1.3: Chạy Stage 2 — GroundingDINO Detection

```python
%cd /content/AIC2026/data_processing/object_detection

from stage2_gdino_detector import run_stage2
from config import STAGE2_OUTPUT
from utils import Timer

with Timer("Stage 2 — GroundingDINO"):
    stage2_results = run_stage2(input_dir="/content/keyframes_test")

print(f"\n📁 Output: {STAGE2_OUTPUT}")
print(f"   Exists: {STAGE2_OUTPUT.exists()}")
```

### Cell 1.4: Lưu kết quả Stage 1+2 vào Drive (backup)

```python
import shutil
backup_dir = "/content/drive/MyDrive/AIC2026_output/intermediate"
!mkdir -p {backup_dir}

from config import STAGE1_OUTPUT, STAGE2_OUTPUT

shutil.copy2(str(STAGE1_OUTPUT), backup_dir)
shutil.copy2(str(STAGE2_OUTPUT), backup_dir)
print(f"✅ Đã backup Stage 1+2 output vào Drive: {backup_dir}")
```

---

## 🔬 Cell 2: Stage 3 — Co-DETR ViT-L LVIS

> ⚠️ **Stage 3 chạy trong conda env riêng** — PHẢI thực thi qua subprocess/shell.
> KHÔNG import trực tiếp trong Python Colab vì sẽ xung đột thư viện.

### Cell 2.1: Cài đặt Miniconda + env codetr

```python
# ====== Cài đặt Miniconda ======
!wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!bash Miniconda3-latest-Linux-x86_64.sh -b -p /content/miniconda

# Chấp nhận ToS
!/content/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
!/content/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
!/content/miniconda/bin/conda config --add channels conda-forge

# Tạo env codetr
!/content/miniconda/bin/conda create -n codetr python=3.9 pip -y

# Thiết lập biến môi trường
%env ENV_PIP=/content/miniconda/envs/codetr/bin/pip
%env ENV_PYTHON=/content/miniconda/envs/codetr/bin/python

print("✅ Đã tạo conda env 'codetr'")
```

### Cell 2.2: Cài đặt thư viện cho env codetr

```python
# PyTorch 1.12.1 + CUDA 11.3
!$ENV_PIP install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# OpenMIM + MMDetection
!$ENV_PIP install openmim
%env ENV_MIM=/content/miniconda/envs/codetr/bin/mim
!$ENV_MIM install mmcv-full==1.5.0
!$ENV_MIM install mmdet==2.25.3

# Fix MMCV version (cần 1.6.0 cho Co-DETR ViT-L)
!$ENV_PIP install mmcv-full==1.6.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

# Cài mmdet lại để đồng bộ
!$ENV_PIP install mmdet==2.25.3

# Các thư viện bổ sung
!$ENV_PIP install einops fairscale fvcore scipy timm "numpy<2" \
    huggingface_hub matplotlib opencv-python-headless

print("✅ Đã cài đặt xong thư viện cho env codetr")
```

### Cell 2.3: Clone Co-DETR repo + Tải checkpoint

```python
# Clone repo Co-DETR
!git clone https://github.com/Sense-X/Co-DETR.git /content/Co-DETR

# Tải checkpoint LVIS từ HuggingFace
!$ENV_PIP install huggingface_hub
!$ENV_PYTHON -c "
import huggingface_hub
huggingface_hub.hf_hub_download(
    repo_id='zongzhuofan/co-detr-vit-large-lvis',
    filename='pytorch_model.pth',
    local_dir='/content/Co-DETR/checkpoints'
)
"
!mv /content/Co-DETR/checkpoints/pytorch_model.pth \
    /content/Co-DETR/checkpoints/pytorch_model_lvis.pth

print("✅ Đã tải checkpoint Co-DETR ViT-L LVIS")
!ls -la /content/Co-DETR/checkpoints/
```

### Cell 2.4: Chạy Stage 3 — Co-DETR Detection

```python
import os

# Đường dẫn output (phải khớp với Stage 4)
stage3_output = "/content/AIC2026/output/object_detection/intermediate/stage3_codetr_detections.json"
os.makedirs(os.path.dirname(stage3_output), exist_ok=True)

# Copy file stage3 vào thư mục Co-DETR để chạy
!cp /content/AIC2026/data_processing/object_detection/stage3_codetr_detector.py /content/Co-DETR/

# Chạy Stage 3 trong env codetr
!cd /content/Co-DETR && MPLBACKEND=Agg $ENV_PYTHON stage3_codetr_detector.py \
    --input /content/keyframes_test \
    --output {stage3_output} \
    --codetr-dir /content/Co-DETR \
    --score-threshold 0.3 \
    --device cuda:0

print(f"\n📁 Output: {stage3_output}")
!ls -la {stage3_output}
```

### Cell 2.5: Backup Stage 3 output vào Drive

```python
import shutil
backup_dir = "/content/drive/MyDrive/AIC2026_output/intermediate"
!mkdir -p {backup_dir}
shutil.copy2(stage3_output, backup_dir)
print(f"✅ Đã backup Stage 3 output vào Drive")
```

---

## 🔗 Cell 3: Stage 4 — IoU Merge (CPU only)

> Stage 4 chạy trên CPU, không cần GPU, chạy trong env nào cũng được.

### Cell 3.1: Chạy Stage 4

```python
%cd /content/AIC2026/data_processing/object_detection

# Reload modules
import sys, importlib
for mod_name in ['config', 'utils', 'stage4_iou_merger']:
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])

from stage4_iou_merger import run_stage4
from config import STAGE4_OUTPUT
from utils import Timer

with Timer("Stage 4 — IoU Merge"):
    documents = run_stage4(video_id="L21_V001")

print(f"\n📁 Output: {STAGE4_OUTPUT}")
```

### Cell 3.2: Kiểm tra kết quả

```python
import json
from config import STAGE4_OUTPUT

# Đọc 3 documents đầu tiên
with open(STAGE4_OUTPUT, "r") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        doc = json.loads(line)
        print(f"\n{'='*60}")
        print(f"Frame: {doc['frame_id']}")
        print(f"Tags ({doc['tag_count']}): {doc['tags']}")
        print(f"Objects ({doc['object_count']}): {doc['object_summary']}")
        print(f"Class counts: {doc['class_counts']}")
        for obj in doc['objects']:
            parent_info = f" → parent={obj['parent_id']}" if obj['parent_id'] >= 0 else ""
            print(f"  [{obj['source']:6s}] {obj['label']:<15s} "
                  f"score={obj['score']:.3f} quad={obj['quadrant']}{parent_info}")
```

### Cell 3.3: Backup kết quả cuối vào Drive

```python
import shutil
final_dir = "/content/drive/MyDrive/AIC2026_output/final"
!mkdir -p {final_dir}

from config import STAGE4_OUTPUT, FINAL_OUTPUT_DIR

# Copy JSONL + report
shutil.copy2(str(STAGE4_OUTPUT), final_dir)
report_path = FINAL_OUTPUT_DIR.parent / "intermediate" / "merge_report.json"
if report_path.exists():
    shutil.copy2(str(report_path), final_dir)

print(f"✅ Đã backup kết quả cuối cùng vào: {final_dir}")
```

---

## 📊 Cell 4: Tổng hợp Benchmark

```python
import json

# Đọc reports từ các stages
from config import STAGE1_OUTPUT, STAGE2_OUTPUT, STAGE3_OUTPUT

reports = []
for path, name in [(STAGE1_OUTPUT, "Stage 1 (RAM++)"),
                    (STAGE2_OUTPUT, "Stage 2 (GDINO)"),
                    (STAGE3_OUTPUT, "Stage 3 (Co-DETR)")]:
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        reports.append({
            "Stage": name,
            "Frames": data.get("total_frames", "?"),
            "Total (s)": data.get("total_time_s", "?"),
            "Per frame (s)": data.get("avg_time_per_frame_s", "?"),
            "FPS": data.get("avg_fps", "?"),
            "VRAM Peak (MB)": data.get("vram_peak_mb", "?"),
            "Precision": data.get("precision", "?"),
        })

# In bảng tổng hợp
print(f"\n{'='*80}")
print(f"📊 BENCHMARK TỔNG HỢP — 307 KEYFRAMES (L21_V001)")
print(f"{'='*80}")
print(f"{'Stage':<20s} {'Frames':>7s} {'Total(s)':>10s} {'Per frame':>10s} "
      f"{'FPS':>8s} {'VRAM(MB)':>10s} {'Prec':>6s}")
print(f"{'-'*80}")
for r in reports:
    print(f"{r['Stage']:<20s} {str(r['Frames']):>7s} {str(r['Total (s)']):>10s} "
          f"{str(r['Per frame (s)']):>10s} {str(r['FPS']):>8s} "
          f"{str(r['VRAM Peak (MB)']):>10s} {r['Precision']:>6s}")
print(f"{'='*80}")

# Ước tính thời gian cho 32.5M frames
for r in reports:
    if isinstance(r['Per frame (s)'], (int, float)):
        est_days = (32_500_000 * r['Per frame (s)']) / 86400
        est_8gpu = est_days / 8
        print(f"\n📐 {r['Stage']}:")
        print(f"   32.5M frames × {r['Per frame (s)']}s = {est_days:.1f} ngày (1 GPU)")
        print(f"   Với 8 GPU: {est_8gpu:.1f} ngày")
```

---

## 📝 Checklist trước khi chạy

- [ ] Đã mount Google Drive
- [ ] Đã clone repo + checkout đúng branch
- [ ] Keyframes đã có tại `/content/keyframes_test/` (307 files)
- [ ] Colab runtime = GPU (T4 hoặc A100)
- [ ] Chạy Cell 1 trước Cell 2 (khác env)
- [ ] **KHÔNG** chạy `pip install -U transformers` ở bất kỳ cell nào

## ⚠️ Xử lý sự cố

| Lỗi | Nguyên nhân | Giải pháp |
|:----|:-----------|:----------|
| `ImportError: apply_chunking_to_forward` | transformers bị nâng cấp | Restart runtime + chạy lại Cell 1.1 |
| `CUDA out of memory` | VRAM không đủ | Restart runtime để giải phóng VRAM |
| `Config file not found` | Chưa clone Co-DETR | Chạy lại Cell 2.3 |
| `Stage X output not found` | Chưa chạy stage trước | Chạy các stages theo thứ tự 1→2→3→4 |
