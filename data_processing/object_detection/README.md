# RAM++ + GroundingDINO Object Detection

This workflow is independent from OCR. It produces object tags, bounding boxes,
object summaries, and counts for keyframes. The JSONL output can be merged into
the database later by `frame_id` / `video_id`.

## Server Env

Create a separate conda env under `/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs`:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/object_detection
bash scripts/setup_object_detection_env.sh
```

Activate it later with:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-object-detection-gpu
```

Expected model layout:

```text
/tmp2/maitanha/vgu/ttn/AIC-Khoa/models/
└── ram_plus_swin_large_14m.pth
```

Download RAM++ checkpoint once if it is not available:

```bash
mkdir -p /tmp2/maitanha/vgu/ttn/AIC-Khoa/models
wget -O /tmp2/maitanha/vgu/ttn/AIC-Khoa/models/ram_plus_swin_large_14m.pth \
  https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth
```

If `recognize-anything` is not available from the server PyPI mirror:

```bash
pip install git+https://github.com/xinyu1205/recognize-anything.git
```

GroundingDINO is loaded from HuggingFace and cached under:

```text
/tmp2/maitanha/vgu/ttn/AIC-Khoa/hf_cache
```

## A5000 Defaults

The default server config is:

```text
configs/object_detection_a5000.yaml
```

Key values:

```text
RAM++:
  image_size: 384
  batch_size: 16
  max_prompt_tags: 25

GroundingDINO:
  model_id: IDEA-Research/grounding-dino-base
  image_max_side: 1280
  box_threshold: 0.30
  text_threshold: 0.25

Postprocess:
  nms_iou_threshold: 0.70
  scene_area_threshold: 0.60
```

## Smoke Test

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/object_detection
bash scripts/run_l22_v012_smoke.sh
```

Override sample size:

```bash
LIMIT=100 bash scripts/run_l22_v012_smoke.sh
```

Visualize smoke-test detections:

```bash
python visualize_detections.py \
  --jsonl ./outputs/object_detection/L22_V012_objects.jsonl \
  --output-dir ./outputs/object_detection/visualized \
  --frames-dir ./keyframe_test/L22_V012 \
  --limit 20
```

## Batch Run

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/object_detection

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-object-detection-gpu

CUDA_VISIBLE_DEVICES=0 \
RAM_CHECKPOINT_PATH=/tmp2/maitanha/vgu/ttn/AIC-Khoa/models/ram_plus_swin_large_14m.pth \
HF_HOME=/tmp2/maitanha/vgu/ttn/AIC-Khoa/hf_cache \
python ram_gdino_pipeline.py \
  --config configs/object_detection_a5000.yaml \
  --input /tmp2/maitanha/vgu/ttn/data/AIC2025/keyframes/Keyframes_L22 \
  --output ./outputs/object_detection \
  --batch \
  --resume \
  --summary-output ./outputs/object_detection/summaries/L22_summary.json
```

If A5000 VRAM is tight, try `--ram-batch-size 8` or `--image-max-side 960`.
If small objects are missed, benchmark `--image-max-side 1536` on a sample first.
