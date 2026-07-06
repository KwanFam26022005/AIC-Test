# RAM++ + LocateAnything Object Detection

Pipeline này độc lập với OCR. Nó chỉ tạo object tags, bounding boxes và object counts cho keyframes. Kết quả JSONL có thể được merge vào database sau, nhưng code trong thư mục này không import hoặc sửa phần OCR.

## Files

```text
Ram_Locate/
├── run_ram_tag_cache.py                 # Phase A: RAM++ -> tag cache JSONL
├── run_locate_detection.py              # Phase B: tag cache -> LocateAnything -> objects JSONL
├── locate_worker.py                     # Persistent official LocateAnything worker
├── tag_filter.py                        # RAM++ tag filtering + LocateAnything prompt builder
├── postprocess.py                       # bbox cleanup, label normalize, NMS, counts
├── ram_locate_common.py                 # shared JSONL/frame/device utilities
├── requirements.txt                     # separate server env deps
└── ram_locate_optimized_pipeline_plan.md
```

## Recommended Server Env

Create a separate env. Do not install this into the OCR env.

```bash
conda create -n aic-ram-locate-gpu python=3.10 -y
conda activate aic-ram-locate-gpu

# Install the CUDA build matching the server first, for example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/object_detection/Ram_Locate
pip install -r requirements.txt
```

If `recognize-anything` is not available from your PyPI mirror:

```bash
pip install git+https://github.com/xinyu1205/recognize-anything.git
```

Model/cache layout example:

```text
/tmp2/maitanha/vgu/ttn/AIC-Khoa/models/
├── ram_plus_swin_large_14m.pth
└── LocateAnything-3B/
```

Download LocateAnything once on the server:

```bash
huggingface-cli download nvidia/LocateAnything-3B \
  --local-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/models/LocateAnything-3B
```

## Phase A - RAM++ Tag Cache

Smoke test one video, 20 frames:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/object_detection/Ram_Locate

CUDA_VISIBLE_DEVICES=0 \
RAM_CHECKPOINT_PATH=/tmp2/maitanha/vgu/ttn/AIC-Khoa/models/ram_plus_swin_large_14m.pth \
python run_ram_tag_cache.py \
  --input ../../../keyframe_test/L22_V012 \
  --output ./outputs/tag_cache/L22_V012_ram_tags.jsonl \
  --video-id L22_V012 \
  --pattern "*.jpg" \
  --limit 20 \
  --ram-batch-size 16 \
  --max-prompt-tags 20 \
  --resume
```

Batch all video folders:

```bash
CUDA_VISIBLE_DEVICES=0 \
RAM_CHECKPOINT_PATH=/tmp2/maitanha/vgu/ttn/AIC-Khoa/models/ram_plus_swin_large_14m.pth \
python run_ram_tag_cache.py \
  --input /tmp2/maitanha/vgu/ttn/data/AIC2025/keyframes/Keyframes_L22 \
  --output ./outputs/tag_cache \
  --batch \
  --pattern "*.jpg" \
  --ram-batch-size 16 \
  --max-prompt-tags 20 \
  --resume
```

## Phase B - LocateAnything Detect/Count

Smoke test using the tag cache from Phase A:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/tmp2/maitanha/vgu/ttn/AIC-Khoa/hf_cache \
python run_locate_detection.py \
  --tag-cache ./outputs/tag_cache/L22_V012_ram_tags.jsonl \
  --output ./outputs/detections/L22_V012_objects.jsonl \
  --frames-dir ../../../keyframe_test/L22_V012 \
  --model-path /tmp2/maitanha/vgu/ttn/AIC-Khoa/models/LocateAnything-3B \
  --dtype fp16 \
  --generation-mode hybrid \
  --image-max-side 1280 \
  --max-new-tokens 1024 \
  --max-prompt-tags 20 \
  --limit 20 \
  --resume \
  --summary-output ./outputs/summaries/L22_V012_locate_summary.json
```

If FlashAttention is installed and compatible, add `--attn-implementation flash_attention_2`. The worker will retry default attention if that load path fails.

Batch all tag cache JSONLs:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/tmp2/maitanha/vgu/ttn/AIC-Khoa/hf_cache \
python run_locate_detection.py \
  --tag-cache ./outputs/tag_cache \
  --output ./outputs/detections \
  --batch \
  --frames-root /tmp2/maitanha/vgu/ttn/data/AIC2025/keyframes/Keyframes_L22 \
  --model-path /tmp2/maitanha/vgu/ttn/AIC-Khoa/models/LocateAnything-3B \
  --dtype fp16 \
  --generation-mode hybrid \
  --image-max-side 1280 \
  --max-new-tokens 1024 \
  --max-prompt-tags 20 \
  --resume \
  --summary-output ./outputs/summaries/locate_batch_summary.json
```

## A5000 Defaults

Use these defaults first:

```text
RAM++:
  ram_batch_size: 16
  ram_image_size: 384
  max_prompt_tags: 20

LocateAnything:
  dtype: fp16
  generation_mode: hybrid
  image_max_side: 1280
  max_new_tokens: 1024
  max_prompt_tags: 20
  worker_count_per_gpu: 1

Postprocess:
  scene_area_threshold: 0.65
  nms_iou_threshold: 0.75
  repeated_box_iou_threshold: 0.98
```

If it is too slow, try `--image-max-side 960` or `--generation-mode fast` on a sample first. If it misses small objects, try `--image-max-side 1536` and `--max-new-tokens 2048`, then re-benchmark VRAM/time.

## Benchmark Protocol

Do not start with 1M frames.

```text
1. Smoke test: 20 frames
2. Benchmark: 1,000 frames
3. Pilot: 10,000 frames
4. Full run by video/shard
```

Runtime estimate:

```text
days = avg_sec_per_frame * 1_000_000 / 86400
cost = avg_sec_per_frame * 1_000_000 / 3600 * hourly_rate
```

## Output Notes

`objects` do not contain model confidence by default because LocateAnything does not expose a calibrated per-box score in the normal text output. Count is computed after:

```text
parse boxes -> clamp -> degenerate filter -> scene filter -> same-label dedupe
-> class-agnostic NMS -> Counter(label)
```

The final JSONL is intentionally independent from OCR and can be merged into the database later by `frame_id` / `video_id`.
