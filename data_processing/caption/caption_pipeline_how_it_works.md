# Caption Pipeline - How It Works

## 1. Mục tiêu

Caption pipeline tạo một lớp text thống nhất cho video search/retrieval từ 4 nguồn:

- Keyframes: ảnh đã extract sẵn theo `keyframe_test/<video_id>/`.
- OCR: chữ đọc được trên từng frame.
- Object detection / scene tags: vật thể, RAM tags, scene tags.
- Audio / ASR: lời nói được align theo timestamp.

Pipeline không chỉ tạo caption cho ảnh. Nó tạo 3 cấp text:

```text
Frame caption -> Shot caption -> Event / TRAKE text -> Search index
```

Trong đó:

- `Frame caption`: mô tả từng keyframe.
- `Shot caption`: mô tả một đoạn video ngắn gồm nhiều frame gần nhau.
- `Event caption`: diễn giải shot thành sự kiện/hành động chính.
- `TRAKE text`: text tối ưu cho truy vấn và retrieval.

---

## 2. Input chính

Config official hiện nằm ở:

```text
data_processing/caption/configs/official/caption_official_l22_v012.yaml
```

Input cho `L22_V012`:

```text
outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs.jsonl
outputs/object_detection/L22_V012_objects.jsonl
outputs/audio/L22_V012/features/audio_features.jsonl
keyframe_test/
```

Pipeline join dữ liệu chủ yếu bằng:

```text
video_id + canonical_frame_id
```

Timeline dùng:

```text
timestamp_sec
```

Không nên join bằng `legacy_frame_id`, tên file rời rạc, hoặc thứ tự dòng nếu đã có `canonical_frame_id`.

---

## 3. Output chính

Official output mặc định:

```text
caption_official/<video_id>/
├── evidence/
│   ├── frame_evidence.jsonl
│   └── shot_evidence.jsonl
├── captions/
│   ├── frame_index.jsonl
│   ├── shot_index.jsonl
│   └── event_step_index.jsonl
├── indexes/
│   └── compact_search_index.jsonl
├── reports/
│   ├── frame_caption_report.json
│   ├── shot_caption_report.json
│   ├── event_step_report.json
│   └── visualization/
│       ├── index.html
│       ├── frame_review.html
│       ├── shot_review.html
│       └── assets/
└── checkpoints/
```

Work/runtime output cho VLM:

```text
caption_official_work/vlm/<video_id>/frame_caption_overrides.jsonl
```

---

## 4. Các phase trong official workflow

Entry point chính:

```text
data_processing/caption/run_official_caption_pipeline.py
```

Workflow official chạy theo 4 nhóm stage lớn.

### Stage 1 - Template bootstrap

Command nội bộ gọi:

```text
run_caption_pipeline.py
```

Mode:

```text
caption_mode=template
shot_caption_mode=template
trake_event_mode=template
```

Mục đích:

- Build evidence baseline.
- Join OCR/object/audio theo frame.
- Group frame thành shot.
- Tạo output đầy đủ bằng template để có baseline an toàn.
- Tạo compact index ban đầu.

Stage này không cần VLM/LLM nặng.

### Stage 2 - VLM initial frame captions

Command nội bộ gọi:

```text
run_vlm_frame_caption_experiment.py
```

Model hiện tại:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

Prompt:

```text
caption_pipeline/prompts/vlm_initial_caption_prompt.txt
```

Mục đích:

- Nhìn ảnh keyframe thật.
- Sinh caption ban đầu tốt hơn template cho một tập frame đại diện.
- Ghi override vào:

```text
caption_official_work/vlm/<video_id>/frame_caption_overrides.jsonl
```

VLM dùng GPU.

### Stage 3 - Prompt-backed caption pipeline

Command nội bộ gọi lại:

```text
run_caption_pipeline.py
```

Mode:

```text
caption_mode=llm
shot_caption_mode=llm
trake_event_mode=llm
```

Model hiện tại:

```text
Qwen/Qwen2.5-7B-Instruct
```

Prompts:

```text
caption_pipeline/prompts/frame_caption_prompt.txt
caption_pipeline/prompts/shot_caption_prompt.txt
caption_pipeline/prompts/trake_event_step_prompt.txt
```

Mục đích:

- Dùng VLM frame overrides làm visual grounding.
- Dùng OCR/object/audio evidence để fuse caption.
- Sinh `frame_index.jsonl`.
- Sinh `shot_index.jsonl`.
- Sinh `event_step_index.jsonl`.
- Rebuild `compact_search_index.jsonl` với caption + TRAKE text.

Stage này dùng text LLM và cần GPU nếu chạy model local bằng transformers.

### Stage 4 - Retrieval export/eval

Command nội bộ gọi:

```text
run_retrieval_export_eval.py
```

Scoring profile official:

```text
route_gated_rrf
```

Mục đích:

- Export/evaluate compact search corpus.
- Kiểm tra retrieval queries.
- Tạo report ở thư mục `eval/`.

Stage này chủ yếu CPU, không cần GPU.

---

## 5. Khái niệm Frame, Shot, Event, TRAKE

### Frame

Một keyframe là một ảnh tại một timestamp.

Frame-level data gồm:

- `canonical_frame_id`
- `timestamp_sec`
- `image_relpath`
- OCR text
- object/scene/RAM tags
- audio gần timestamp
- frame caption

Frame dùng để kiểm tra:

```text
Ảnh này được hiểu đúng chưa?
```

### Shot

Một shot là một đoạn thời gian ngắn gồm nhiều frame liên tiếp có ngữ cảnh gần nhau.

Shot-level data gồm:

- start/end time
- representative frames
- merged OCR
- merged audio
- merged object counts
- merged scene tags
- shot caption

Shot dùng để kiểm tra:

```text
Đoạn video này được tổng hợp đúng chưa?
```

### Event caption

Event caption diễn giải shot thành sự kiện/hành động chính.

Nó trả lời:

```text
Trong đoạn này đang xảy ra việc gì?
```

### TRAKE text

TRAKE text là câu giàu thông tin hơn để phục vụ retrieval.

Nó thường cố gắng chứa:

```text
actor + action + object/topic + scene/context
```

Nó trả lời:

```text
Nếu người dùng search bằng ngôn ngữ tự nhiên, đoạn này nên được index bằng câu nào?
```

---

## 6. Runtime models và môi trường

Official config hiện dùng:

```yaml
models:
  vlm:
    provider: qwen2_5_vl
    model_name: Qwen/Qwen2.5-VL-7B-Instruct
    dtype: bfloat16
    device_map: auto
    attn_implementation: flash_attention_2

  text_llm:
    provider: transformers
    model_name: Qwen/Qwen2.5-7B-Instruct
    dtype: bfloat16
    device_map: auto
    attn_implementation: flash_attention_2
```

Env example:

```text
data_processing/caption/configs/official/caption_official.env.example
```

Conda env đã dùng trong quá trình test:

```text
/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm
```

Các package quan trọng:

- `torch`
- `transformers`
- `accelerate`
- `qwen-vl-utils`
- `flash-attn`
- `PyYAML`

---

## 7. Cách chạy official pipeline

Ví dụ cho `L22_V012`:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

CUDA_VISIBLE_DEVICES=1 \
PYTHONIOENCODING=utf-8 \
python run_official_caption_pipeline.py \
  --config configs/official/caption_official_l22_v012.yaml \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_official.env \
  --env-override \
  --no-resume
```

Nếu chỉ muốn chạy lại một số phần:

```bash
--skip-bootstrap
--skip-vlm
--skip-llm
--skip-retrieval
```

Ví dụ chỉ chạy retrieval sau khi caption đã có:

```bash
python run_official_caption_pipeline.py \
  --config configs/official/caption_official_l22_v012.yaml \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_official.env \
  --env-override \
  --skip-bootstrap \
  --skip-vlm \
  --skip-llm
```

---

## 8. Checkpoint và resume

Text LLM runtime có checkpoint ở:

```text
caption_official/<video_id>/checkpoints/
```

Các stage LLM có thể resume nếu không dùng `--no-resume`.

Dùng `--no-resume` khi:

- Prompt thay đổi.
- Schema output thay đổi.
- Muốn chạy sạch lại từ đầu.

Không dùng `--no-resume` khi:

- Job bị ngắt giữa chừng và muốn tiếp tục.
- Chỉ bị lỗi tạm thời do GPU/memory.

---

## 9. Visualization review

Generate visualization:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_caption_visualize.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_official \
  --keyframes-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --image-mode copy \
  --max-frames-per-shot 3
```

Mở bằng HTTP server:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_official/L22_V012/reports/visualization
python -m http.server 8899
```

Các trang:

```text
index.html
frame_review.html
shot_review.html
```

### Tab Frames

Mục đích:

```text
Kiểm tra từng ảnh/keyframe có được hiểu đúng chưa.
```

Xem:

- image
- frame id
- timestamp
- shot id
- frame caption
- OCR
- objects
- scene/RAM tags
- audio/ASR gần timestamp

Nếu frame caption sai, kiểm lại ảnh và evidence trong modal frame detail.

### Tab Shots

Mục đích:

```text
Kiểm tra từng đoạn video đã được tổng hợp và index tốt chưa.
```

Xem:

- representative images
- shot caption
- event caption
- TRAKE text
- merged OCR
- merged audio
- merged objects
- merged scene tags
- search/quality fields

Nếu shot caption hoặc TRAKE text sai, kiểm lại các frame đại diện và evidence tổng hợp.

---

## 10. Acceptance checks

Với `L22_V012`, một run tốt thường có:

```text
Frames: 283
Shots: 102
Index docs: 385
Frame captions: 283/283
Shot captions: 102/102
Event steps: 102/102
Warnings: 0 hoặc không có blocking warnings
```

Sau khi generate visualization, kiểm ảnh:

```bash
VIS=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_official/L22_V012/reports/visualization

grep -o '<img src="assets/' "$VIS/frame_review.html" | wc -l
grep -o '<img src="assets/' "$VIS/shot_review.html" | wc -l
ls -lh "$VIS/assets" | head
```

---

## 11. File code quan trọng

Entry points:

```text
run_official_caption_pipeline.py
run_caption_pipeline.py
run_caption_visualize.py
run_retrieval_export_eval.py
run_vlm_frame_caption_experiment.py
```

Core modules:

```text
caption_pipeline/load_inputs.py
caption_pipeline/evidence_alignment.py
caption_pipeline/evidence_builder.py
caption_pipeline/shot_grouper.py
caption_pipeline/frame_caption_fuser.py
caption_pipeline/shot_captioner.py
caption_pipeline/trake_event_builder.py
caption_pipeline/compact_index.py
caption_pipeline/visualization.py
```

Runtime:

```text
caption_pipeline/runtime/text_llm.py
caption_pipeline/runtime/json_output.py
caption_pipeline/runtime/checkpoint.py
caption_pipeline/runtime/prompt_loader.py
caption_pipeline/runtime/env_loader.py
```

Prompts:

```text
caption_pipeline/prompts/vlm_initial_caption_prompt.txt
caption_pipeline/prompts/frame_caption_prompt.txt
caption_pipeline/prompts/shot_caption_prompt.txt
caption_pipeline/prompts/trake_event_step_prompt.txt
```

---

## 12. Ghi chú thiết kế

- Evidence alignment phải ổn trước khi tin caption.
- `canonical_frame_id` là khóa join quan trọng nhất.
- OCR review text không nên dùng mặc định cho caption.
- Audio chỉ dùng segment `usable_for_caption=true`.
- VLM giúp frame caption bám ảnh hơn.
- Text LLM giúp fuse evidence thành caption/event/search text.
- TRAKE text là lớp quan trọng nhất cho retrieval ở cấp shot/event.
- Visualization là công cụ review, không phải output search cuối cùng.
