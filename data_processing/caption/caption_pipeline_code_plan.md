# Caption Pipeline Code Plan

## 0. Mục tiêu

Plan này thay thế hướng triển khai chung trong `caption_recap_fusecap_plan.md` bằng một kế hoạch code cụ thể dựa trên output mới nhất trong `outputs/`.

Mục tiêu trước mắt:

- Không gọi LLM/VLM ngay.
- Viết pipeline chuẩn hóa và align evidence từ OCR, object detection và audio.
- Sinh `frame_evidence.jsonl`, `shot_evidence.jsonl`, `compact_search_index.jsonl`.
- Tạo validation report để chắc chắn caption fuser sau này không join sai frame, không dùng text nhiễu, không thiếu timestamp.

Kiến trúc FuseCap/ReCap/TRAKE vẫn giữ, nhưng thứ tự thực hiện phải bắt đầu từ evidence alignment.

---

## 1. Audit output mới nhất

Output đã kiểm tra:

```text
outputs/
├── audio/L22_V012/
├── object_detection/
└── ocr_vlm_pipeline_v2/L22_V012/
```

### 1.1. OCR

File:

```text
outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs.jsonl
outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_quality_summary.json
```

Schema thực tế:

```json
{
  "schema_version": "ocr_vlm_pipeline_v2_es_2",
  "document_id": "ocr:L22_V012_001",
  "video_id": "L22_V012",
  "frame_id": "L22_V012_001",
  "canonical_frame_id": "L22_V012_001",
  "legacy_frame_id": "001",
  "frame_name": "001",
  "keyframe_idx": 1,
  "source_frame_idx": 0,
  "timestamp_sec": 0.0,
  "timestamp_source": "keyframe_map_csv",
  "image_relpath": "L22_V012/001.jpg",
  "ocr_text_clean": "giây",
  "ocr_text_search": "giây",
  "ocr_terms": ["giay"]
}
```

Quality summary:

```text
num_frames = 283
num_frames_missing_timestamp = 0
num_frames_missing_canonical_frame_id = 0
num_docs_out_of_order = 0
num_search_duplicates = 0
num_need_review_lines_in_primary_search = 0
num_frames_with_clean_text = 263
num_frames_with_review_text = 189
```

Caption rule:

- Dùng `ocr_text_search` làm OCR evidence chính.
- Không dùng `ocr_text_review` mặc định.
- Có thể giữ `ocr_text_review` trong debug/evidence audit.

### 1.2. Object detection

File:

```text
outputs/object_detection/L22_V012_objects.jsonl
outputs/object_detection/summaries/L22_V012_summary.json
```

Schema thực tế:

```json
{
  "schema_version": "ram_gdino_object_detection_v1_1",
  "video_id": "L22_V012",
  "frame_id": "L22_V012_001",
  "canonical_frame_id": "L22_V012_001",
  "frame_name": "001",
  "keyframe_idx": 1,
  "source_frame_idx": 0,
  "timestamp_sec": 0.0,
  "timestamp_source": "keyframe_map_csv",
  "image_relpath": "L22_V012/001.jpg",
  "object_tags": [],
  "scene_tags": ["city", "sky", "sun", "water"],
  "object_counts_normalized": {},
  "important_objects": [],
  "object_text": "",
  "all_object_text": "city sky sun water boat city ..."
}
```

Quality summary:

```text
num_frames = 283
processed = 283
errors = 0
num_frames_missing_timestamp = 0
num_objects_missing_area_ratio = 0
num_objects_missing_position = 0
num_scene_labels_in_counts = 7
```

Join audit:

```text
OCR rows = 283
Object rows = 283
Joined by canonical_frame_id = 283
Timestamp mismatch = 0
```

Known guardrail:

- Object summary still reports `num_scene_labels_in_counts = 7`.
- Labels involved: `building`, `flower`, `person`, `tree`.
- Caption evidence builder must not blindly trust scene/count separation.
- Use `object_counts_normalized` as primary, but filter any label also present in `scene_tags`.

### 1.3. Audio

File:

```text
outputs/audio/L22_V012/features/audio_features.jsonl
outputs/audio/L22_V012/features/audio_quality_summary.json
```

Schema thực tế:

```json
{
  "schema_version": "audio_feature_v1",
  "feature_id": "audfeat_L22_V012_000001_001",
  "video_id": "L22_V012",
  "start_sec": 0.0,
  "end_sec": 1.71,
  "time_window": [0.0, 1.71],
  "clean_transcript": "...",
  "caption_text": "",
  "search_text": "...",
  "usable_for_caption": false,
  "exclude_from_caption_reason": "boilerplate",
  "quality_level": "bad"
}
```

Quality summary:

```text
num_audio_features = 226
num_caption_usable = 217
num_boilerplate_filtered = 9
num_duplicate_filtered = 3
timeline_sorted = true
num_out_of_order = 0
```

Caption rule:

- Dùng `caption_text` nếu `usable_for_caption=true`.
- Nếu output cũ chưa có `caption_text`, fallback `clean_transcript` only when `usable_for_caption=true`.
- Không dùng `search_text` để tạo caption, chỉ dùng cho compact search/audit.

---

## 2. Kết luận schema dùng cho caption

Caption pipeline join theo:

```text
video_id + canonical_frame_id
```

Timeline dùng:

```text
timestamp_sec
```

Không dùng:

- OCR `legacy_frame_id` để join.
- Object `frame_idx` để suy luận timestamp.
- Raw ASR `asr_segments.jsonl` cho caption.
- OCR `ocr_text_review` trong primary caption.

Fallback cho output cũ:

- Nếu không có `canonical_frame_id`, dùng `frame_id` nếu đã có prefix `video_id_`.
- Nếu OCR `frame_id` dạng `001`, tạo `canonical_frame_id = f"{video_id}_{frame_id}"`.
- Nếu object thiếu `timestamp_sec`, dùng keyframe CSV map.
- Nếu audio thiếu `caption_text`, dùng `clean_transcript` khi `usable_for_caption=true`.

---

## 3. Folder/code layout

Tạo package:

```text
data_processing/caption/
├── caption_pipeline_code_plan.md
├── run_caption_pipeline.py
└── caption_pipeline/
    ├── __init__.py
    ├── config.py
    ├── io_utils.py
    ├── schemas.py
    ├── load_inputs.py
    ├── evidence_alignment.py
    ├── evidence_builder.py
    ├── shot_grouper.py
    ├── compact_index.py
    ├── validation.py
    └── text_utils.py
```

Không cần thêm dependency nặng ở Phase 0.

---

## 4. CLI đề xuất

File:

```text
run_caption_pipeline.py
```

CLI:

```bash
python run_caption_pipeline.py \
  --video-id L22_V012 \
  --ocr-jsonl outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs.jsonl \
  --object-jsonl outputs/object_detection/L22_V012_objects.jsonl \
  --audio-features outputs/audio/L22_V012/features/audio_features.jsonl \
  --keyframe-map keyframe_test \
  --output-dir outputs/caption \
  --frame-window-before 5 \
  --frame-window-after 5 \
  --shot-max-gap-sec 5 \
  --shot-max-duration-sec 20
```

Outputs:

```text
outputs/caption/L22_V012/
├── evidence/
│   ├── frame_evidence.jsonl
│   └── shot_evidence.jsonl
├── indexes/
│   └── compact_search_index.jsonl
└── reports/
    ├── evidence_alignment_report.json
    └── evidence_alignment_report.md
```

---

## 5. Internal schema

### 5.1. FrameEvidence

```json
{
  "schema_version": "caption_frame_evidence_v1",
  "video_id": "L22_V012",
  "frame_id": "L22_V012_001",
  "canonical_frame_id": "L22_V012_001",
  "frame_name": "001",
  "keyframe_idx": 1,
  "source_frame_idx": 0,
  "timestamp_sec": 0.0,
  "image_path": "/tmp2/.../001.jpg",
  "image_relpath": "L22_V012/001.jpg",
  "object_evidence": {
    "object_counts": {},
    "object_tags": [],
    "scene_tags": ["city", "sky", "sun", "water"],
    "ram_tags": ["boat", "city", "sunset", "water"],
    "important_objects": [],
    "object_text": "",
    "scene_text": "city sky sun water",
    "all_object_text": "..."
  },
  "ocr_evidence": {
    "ocr_text": "giây",
    "ocr_terms": ["giay"],
    "has_clean_text": true,
    "review_text": "",
    "review_used": false
  },
  "audio_evidence": {
    "window": [0.0, 5.0],
    "segments": [],
    "audio_text": "",
    "num_segments": 0
  },
  "quality": {
    "has_object": false,
    "has_scene_tags": true,
    "has_ocr": true,
    "has_audio": false,
    "alignment_warnings": []
  },
  "frame_search_text": "giây city sky sun water ..."
}
```

### 5.2. ShotEvidence

```json
{
  "schema_version": "caption_shot_evidence_v1",
  "video_id": "L22_V012",
  "shot_id": "L22_V012_shot_0001",
  "start_sec": 0.0,
  "end_sec": 8.56667,
  "representative_frame_ids": [
    "L22_V012_001",
    "L22_V012_002",
    "L22_V012_003"
  ],
  "frame_count": 3,
  "merged_object_counts": {
    "person": 2,
    "screen": 1
  },
  "merged_scene_tags": ["city", "sky", "screen"],
  "merged_ocr_text": "giây ...",
  "merged_audio_text": "...",
  "shot_search_text": "..."
}
```

### 5.3. CompactSearchDoc

```json
{
  "schema_version": "caption_compact_search_v1",
  "unit_type": "frame",
  "unit_id": "L22_V012_001",
  "video_id": "L22_V012",
  "frame_id": "L22_V012_001",
  "timestamp_sec": 0.0,
  "start_sec": 0.0,
  "end_sec": 0.0,
  "ocr_text": "giây",
  "audio_text": "",
  "object_text": "",
  "scene_text": "city sky sun water",
  "caption_text": "",
  "trake_text": "",
  "all_text": "giây city sky sun water ..."
}
```

---

## 6. Loader requirements

### 6.1. OCR loader

Input:

```text
*_ocr_es_docs.jsonl
```

Use fields:

- `video_id`
- `canonical_frame_id`
- `frame_id`
- `frame_name`
- `keyframe_idx`
- `source_frame_idx`
- `timestamp_sec`
- `image_path`
- `image_relpath`
- `ocr_text_search`
- `ocr_text_clean`
- `ocr_terms`
- `quality`

Rules:

- `ocr_text = ocr_text_search`.
- `review_text = ocr_text_review`, but `review_used=false`.
- Reject OCR primary text if `quality.num_need_review_lines_in_primary_search > 0`.
- Warn if missing `timestamp_sec` or `canonical_frame_id`.

### 6.2. Object loader

Input:

```text
*_objects.jsonl
```

Use fields:

- `canonical_frame_id`
- `timestamp_sec`
- `object_counts_normalized`
- `object_count_items`
- `object_tags`
- `scene_tags`
- `ram_tags`
- `important_objects`
- `object_text`
- `scene_text`
- `all_object_text`
- `objects`

Guardrails:

- Remove any object count label also present in `scene_tags`.
- Drop empty/zero counts.
- For caption evidence, cap objects by:
  - count desc
  - object area/confidence when available
  - `max_objects`
- Keep `scene_tags` separately from countable objects.

### 6.3. Audio loader

Input:

```text
audio_features.jsonl
```

Use fields:

- `start_sec`
- `end_sec`
- `caption_text`
- `clean_transcript`
- `search_text`
- `usable_for_caption`
- `exclude_from_caption_reason`
- `quality_level`

Rules:

- Caption alignment only uses rows with `usable_for_caption=true`.
- Text is `caption_text` if non-empty, otherwise fallback `clean_transcript`.
- Keep `search_text` only for `compact_search_index`, not for caption fuser.

---

## 7. Evidence alignment

### 7.1. Frame join

Primary frame list should come from object detection if present, because object output has all keyframes and image paths.

Join order:

```text
object frames
  left join OCR by canonical_frame_id
  attach audio by timestamp window
```

If object is missing for a frame but OCR exists, Phase 0 can optionally include OCR-only frame with warning:

```text
include_ocr_only_frames: false initially
```

### 7.2. Timestamp validation

For each joined frame:

- OCR timestamp and object timestamp must differ by <= `0.001`.
- If mismatch, keep object timestamp as canonical and report warning.
- If both missing, load keyframe map CSV.

### 7.3. Audio window

Config:

```yaml
audio_alignment:
  frame_window_sec_before: 5.0
  frame_window_sec_after: 5.0
  max_audio_chars_frame: 700
  max_audio_segments_frame: 5
```

Overlap rule:

```text
audio.end_sec >= frame_ts - before
and audio.start_sec <= frame_ts + after
```

Sort selected audio by:

```text
start_sec, end_sec, feature_id
```

Then concatenate `caption_text`.

---

## 8. Evidence builder rules

Config defaults:

```yaml
evidence_builder:
  max_objects: 10
  max_object_tags: 20
  max_scene_tags: 20
  max_ocr_chars: 600
  max_audio_chars_frame: 700
  max_audio_chars_shot: 1500
  keep_object_positions: true
  include_ocr_review: false
  include_audio_search_text_for_caption: false
```

Frame text fields:

```text
frame_search_text =
  ocr_text
  + audio_text
  + object_text
  + scene_text
  + ram_tag_text
```

Do not include review text in `frame_search_text` unless explicitly enabled.

---

## 9. Shot grouping

Initial method:

```yaml
shot_grouping:
  method: timestamp_gap
  max_gap_sec: 5.0
  max_shot_duration_sec: 20.0
  max_representative_frames: 3
  representative_strategy: first_middle_last
```

Algorithm:

- Sort frames by `timestamp_sec`.
- Start new shot if:
  - gap from previous frame > `max_gap_sec`
  - current shot duration would exceed `max_shot_duration_sec`
- Representative frames:
  - 1 frame: first
  - 2 frames: first, last
  - >=3 frames: first, middle, last

Merged object counts:

- Sum countable object labels across frames.
- Cap extremely repeated labels only in text rendering, not in raw counts.

Merged OCR/audio:

- Deduplicate normalized text snippets.
- Preserve chronological order.

---

## 10. Validation report

Output:

```text
evidence_alignment_report.json
evidence_alignment_report.md
```

Metrics:

```json
{
  "video_id": "L22_V012",
  "num_object_frames": 283,
  "num_ocr_frames": 283,
  "num_joined_frames": 283,
  "num_frames_missing_ocr": 0,
  "num_frames_missing_object": 0,
  "num_timestamp_mismatch": 0,
  "num_audio_features": 226,
  "num_audio_features_usable": 217,
  "num_frames_with_ocr": 263,
  "num_frames_with_audio": 100,
  "num_frames_with_objects": 200,
  "num_scene_labels_removed_from_counts": 7,
  "num_frame_evidence": 283,
  "num_shots": 120,
  "warnings": []
}
```

Acceptance for current `L22_V012`:

- `num_joined_frames = 283`
- `num_timestamp_mismatch = 0`
- `num_frames_missing_ocr = 0`
- `num_frame_evidence = 283`
- `audio_features.timeline_sorted = true`
- `ocr.num_docs_out_of_order = 0`
- `object.total_errors = 0`

---

## 11. Compact search baseline

Before LLM captioning, create a useful search index from existing evidence:

Frame docs:

- `unit_type = frame`
- `unit_id = canonical_frame_id`
- `timestamp_sec = frame timestamp`

Shot docs:

- `unit_type = shot`
- `unit_id = shot_id`
- `start_sec`, `end_sec`

Text fields:

- `ocr_text`
- `audio_text`
- `object_text`
- `scene_text`
- `all_text`

This is the first testable output for Elasticsearch before caption generation.

---

## 12. Implementation phases

### Phase 0 - Evidence alignment baseline

Files:

```text
caption_pipeline/io_utils.py
caption_pipeline/load_inputs.py
caption_pipeline/evidence_alignment.py
caption_pipeline/evidence_builder.py
caption_pipeline/validation.py
run_caption_pipeline.py
```

Deliverables:

```text
frame_evidence.jsonl
evidence_alignment_report.json
evidence_alignment_report.md
```

### Phase 1 - Shot grouping and compact index

Files:

```text
caption_pipeline/shot_grouper.py
caption_pipeline/compact_index.py
```

Deliverables:

```text
shot_evidence.jsonl
compact_search_index.jsonl
```

### Phase 2 - Text-only frame caption fuser

Only start after Phase 0/1 validation is clean.

Files:

```text
caption_pipeline/frame_caption_fuser.py
caption_pipeline/prompts/frame_caption_prompt.txt
```

Deliverables:

```text
frame_index.jsonl
```

### Phase 3 - ReCap shot captioner

Files:

```text
caption_pipeline/recap_captioner.py
caption_pipeline/prompts/recap_caption_prompt.txt
```

Deliverables:

```text
shot_index.jsonl
```

### Phase 4 - TRAKE event-step builder

Files:

```text
caption_pipeline/trake_event_builder.py
caption_pipeline/prompts/trake_event_step_prompt.txt
```

Deliverables:

```text
event_step_index.jsonl
```

---

## 13. Code acceptance checklist

Phase 0/1 done when:

- `run_caption_pipeline.py` runs for `L22_V012` without model dependencies.
- `frame_evidence.jsonl` has 283 rows.
- Every frame evidence has:
  - `canonical_frame_id`
  - `timestamp_sec`
  - `image_relpath`
  - `object_evidence`
  - `ocr_evidence`
  - `audio_evidence`
- OCR/object join is 283/283.
- Timestamp mismatch is 0.
- Audio windows use only `usable_for_caption=true`.
- Scene labels found in object counts are removed or reported.
- `compact_search_index.jsonl` exists and has frame + shot docs.
- Validation report has no blocking errors.

---

## 14. Notes before coding

- Do not write LLM prompts first.
- Do not call VLM or image captioner in Phase 0.
- Use structured JSONL readers/writers.
- Keep schema additive and backward-compatible where cheap.
- All downstream caption/search code should prefer normalized fields:
  - OCR: `ocr_text_search`
  - Object: `object_counts_normalized`, `important_objects`, `all_object_text`
  - Audio: `caption_text`
  - Join: `canonical_frame_id`
  - Time: `timestamp_sec`

