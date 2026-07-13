# Audio Extraction Code Fix Plan

Mục tiêu của plan này là sửa triệt để các lỗi trong `data_processing/audio_extraction/` để output audio trở thành nguồn evidence ổn định cho caption fusion, temporal retrieval và TRAKE-style alignment.

Scope chính:

- Không thay đổi model ASR ở bước đầu.
- Không chạy lại toàn bộ video nếu chưa cần.
- Ưu tiên sửa schema, timeline order, quality gate, resume safety và caption-ready output.
- Downstream caption pipeline chỉ nên đọc `audio_features.jsonl`, không đọc trực tiếp `asr_segments.jsonl`.

---

## 1. Hiện trạng lỗi cần sửa

### 1.1. Manifest status gây hiểu nhầm

File thực tế:

- `outputs/audio/L22_V012/manifests/video_manifest.jsonl`
- `outputs/audio/L22_V012/manifests/asr_job_manifest.jsonl`

Vấn đề:

- `video_manifest.jsonl` vẫn ghi `status: pending` dù probe video đã xong.
- `asr_job_manifest.jsonl` vẫn ghi mọi job là `status: pending` dù ASR đã sinh output.
- Downstream hoặc reviewer có thể hiểu nhầm pipeline chưa chạy.

Code liên quan:

- `audio_pipeline/pipeline.py`
- `audio_pipeline/vad.py`

### 1.2. Raw ASR output không theo timeline

Vấn đề:

- `run_asr_jobs()` sort pending jobs theo duration để tối ưu tốc độ.
- `asr_segments.jsonl` được append theo thứ tự chạy job, không theo `start_sec`.
- Hiện output thực tế có nhiều đoạn đảo timeline.
- Nếu ReCap/memory đọc raw ASR theo dòng thì context sẽ sai.

Code liên quan:

- `audio_pipeline/asr.py`
- `audio_pipeline/features.py`

### 1.3. Quality gate để lọt boilerplate ASR

Vấn đề:

- Nhiều câu như `Hãy subscribe...`, `không bỏ lỡ...`, `Amara.org`, `Cảm ơn quý vị đã theo dõi` vẫn được đánh `good`.
- Các câu này không phải nội dung video hữu ích cho caption retrieval.
- Nếu đưa vào caption fuser sẽ gây nhiễu hoặc hallucination topic.

Code liên quan:

- `audio_pipeline/quality.py`
- `audio_pipeline/text.py`
- `configs/audio_pipeline_a5000.yaml`

### 1.4. Resume chưa an toàn

Vấn đề:

- Resume hiện skip job chỉ theo `job_id`.
- Nếu đổi VAD, model, language, beam size hoặc audio file mà `job_id` vẫn giống, pipeline có thể dùng nhầm ASR cũ.

Code liên quan:

- `audio_pipeline/asr.py`
- `audio_pipeline/vad.py`

### 1.5. Output chưa đủ caption-ready

Vấn đề:

- `audio_features.jsonl` có `clean_transcript`, `summary`, `keywords`, `usable_for_caption`, nhưng chưa tách rõ:
  - text dành cho caption
  - text dành cho search
  - lý do loại khỏi caption
  - time window chuẩn

Code liên quan:

- `audio_pipeline/features.py`

---

## 2. Thiết kế output sau khi sửa

### 2.1. `video_manifest.jsonl`

Đổi status thành trạng thái thật:

```json
{
  "schema_version": "video_manifest_v1",
  "video_id": "L22_V012",
  "video_path": ".../L22_V012.mp4",
  "duration_sec": 1204.837,
  "fps": 30.0,
  "width": 1280,
  "height": 720,
  "has_audio": true,
  "audio_codec": "aac",
  "audio_sample_rate": 44100,
  "audio_channels": 2,
  "status": "probed",
  "created_at": "..."
}
```

Allowed statuses:

- `probed`
- `no_audio`
- `probe_failed`

### 2.2. `asr_job_manifest.jsonl`

Job manifest là kế hoạch chạy, không phải result. Đổi `status` từ `pending` sang `planned`, hoặc bỏ `status`.

Thêm `job_fingerprint`.

```json
{
  "job_id": "asrjob_L22_V012_000001",
  "video_id": "L22_V012",
  "audio_path": ".../L22_V012.wav",
  "start_sec": 0.0,
  "end_sec": 1.71,
  "duration_sec": 1.71,
  "segment_type": "speech",
  "priority": 1,
  "status": "planned",
  "job_fingerprint": "sha1:..."
}
```

### 2.3. `asr_segments.jsonl`

Sort theo timeline trước khi ghi final file.

Thêm fingerprint để resume an toàn.

```json
{
  "schema_version": "asr_segment_v1",
  "asr_segment_id": "asrseg_L22_V012_000001_001",
  "job_id": "asrjob_L22_V012_000001",
  "job_fingerprint": "sha1:...",
  "video_id": "L22_V012",
  "start_sec": 0.0,
  "end_sec": 1.71,
  "duration_sec": 1.71,
  "raw_text": "...",
  "language": "vi",
  "status": "success"
}
```

### 2.4. `asr_quality_report.jsonl`

Thêm `reasons` để audit.

```json
{
  "schema_version": "asr_quality_v1",
  "asr_segment_id": "asrseg_L22_V012_000001_001",
  "job_id": "asrjob_L22_V012_000001",
  "video_id": "L22_V012",
  "start_sec": 0.0,
  "end_sec": 1.71,
  "quality": {
    "has_text": true,
    "text_length": 70,
    "token_count": 15,
    "repetition_ratio": 0.0,
    "is_repeated": false,
    "is_too_short": false,
    "is_duration_too_short": false,
    "is_boilerplate": true,
    "is_duplicate_transcript": true,
    "usable_for_caption": false,
    "need_fallback": true,
    "quality_level": "bad",
    "reasons": ["boilerplate", "duplicate_transcript"]
  }
}
```

### 2.5. `audio_features.jsonl`

Đây là file downstream caption nên dùng.

```json
{
  "schema_version": "audio_feature_v1",
  "feature_id": "audfeat_L22_V012_000001_001",
  "modality": "audio",
  "video_id": "L22_V012",
  "start_sec": 0.0,
  "end_sec": 1.71,
  "duration_sec": 1.71,
  "time_window": [0.0, 1.71],
  "raw_transcript": "...",
  "clean_transcript": "...",
  "caption_text": "",
  "search_text": "...",
  "summary": "...",
  "keywords": ["..."],
  "usable_for_caption": false,
  "exclude_from_caption_reason": "boilerplate; duplicate_transcript",
  "quality_level": "bad"
}
```

Rule:

- `caption_text` rỗng nếu `usable_for_caption=false`.
- `search_text` có thể giữ transcript sạch nếu muốn audit/search rộng, nhưng caption không dùng.
- Caption evidence builder chỉ đọc `caption_text` hoặc `clean_transcript` khi `usable_for_caption=true`.

---

## 3. Các thay đổi code cụ thể

### Phase 1 - Sửa manifest status

File: `audio_pipeline/pipeline.py`

Việc cần làm:

- Trong `create_video_manifest()`:
  - đổi `status: "pending"` thành `status: "probed"` khi `has_audio=true`.
  - giữ `status: "no_audio"` khi không có audio.
  - nếu `probe_video()` lỗi thì ghi row `probe_failed` thay vì crash toàn bộ batch, nếu muốn batch robust.

File: `audio_pipeline/vad.py`

Việc cần làm:

- Trong `build_asr_jobs()`:
  - đổi `status: "pending"` thành `status: "planned"`.
  - thêm `job_fingerprint`.

Acceptance:

- Không còn `pending` trong manifest sau khi pipeline hoàn tất.
- `planned` chỉ xuất hiện ở job manifest và được hiểu là plan, không phải result.

### Phase 2 - Sort timeline cho output

File: `audio_pipeline/asr.py`

Việc cần làm:

- Giữ `sort_jobs_by_duration` nếu muốn tối ưu inference.
- Sau khi chạy xong, load lại `asr_segments`, sort theo:

```python
(video_id, start_sec, end_sec, asr_segment_id)
```

- Ghi lại `asr_segments.jsonl` đã sort.

File: `audio_pipeline/features.py`

Việc cần làm:

- Sort `quality_reports` theo timeline.
- `build_audio_features()` đã sort ở cuối, giữ lại và kiểm tra ổn định.

Acceptance:

- `asr_segments.jsonl`, `asr_quality_report.jsonl`, `audio_features.jsonl` đều monotonic theo `start_sec` trong từng `video_id`.

### Phase 3 - Boilerplate và duplicate transcript gate

File: `audio_pipeline/text.py`

Thêm helper:

```python
def normalize_for_quality(text: str) -> str:
    ...

def is_boilerplate_transcript(text: str, patterns: list[str]) -> bool:
    ...
```

File: `audio_pipeline/quality.py`

Việc cần làm:

- Mở rộng `score_quality()` để nhận optional context:

```python
def score_quality(clean_text, duration_sec, cfg, duplicate_count=1):
    ...
```

- Thêm các rule:
  - boilerplate pattern match
  - duplicate transcript count vượt ngưỡng
  - text quá ngắn
  - duration quá ngắn
  - repetition cao

File: `audio_pipeline/features.py`

Việc cần làm:

- Trước khi score từng row, tính duplicate count theo normalized transcript trong cùng video.
- Truyền duplicate count vào `score_quality`.

File: `configs/audio_pipeline_a5000.yaml`

Thêm:

```yaml
quality_gate:
  boilerplate_patterns:
    - "subscribe"
    - "đăng ký"
    - "không bỏ lỡ"
    - "cảm ơn quý vị đã theo dõi"
    - "amara.org"
  duplicate_transcript_min_count: 3
  duplicate_transcript_min_gap_sec: 60
```

Acceptance:

- Các câu subscribe/Amara/outro có `usable_for_caption=false`.
- Quality report có `reasons`.

### Phase 4 - Resume bằng fingerprint

File: `audio_pipeline/vad.py`

Việc cần làm:

- Khi tạo job, thêm fingerprint.
- Fingerprint nên dựa trên dữ liệu ổn định:

```text
video_id
audio_path
start_sec
end_sec
duration_sec
segment_type
asr.model/model_path
asr.language
asr.beam_size
asr.temperature
asr.condition_on_previous_text
asr.vad_filter
```

File: `audio_pipeline/asr.py`

Việc cần làm:

- `done_job_ids` đổi thành map:

```python
done_jobs[(job_id, job_fingerprint)] = row
```

- Chỉ skip nếu cả `job_id` và `job_fingerprint` khớp.
- Ghi `job_fingerprint` vào mỗi ASR row.

Acceptance:

- Đổi config ASR/VAD không dùng nhầm output cũ.
- Chạy lại pipeline không duplicate rows nếu config không đổi.

### Phase 5 - Caption-ready fields

File: `audio_pipeline/features.py`

Việc cần làm:

- Thêm `time_window`.
- Thêm `caption_text`.
- Thêm `search_text`.
- Thêm `exclude_from_caption_reason`.

Logic:

```python
if quality["usable_for_caption"]:
    caption_text = clean_text
else:
    caption_text = ""

search_text = clean_text
exclude_from_caption_reason = "; ".join(quality["reasons"])
```

Acceptance:

- Caption pipeline không cần tự suy luận quality.
- Chỉ cần filter `usable_for_caption=true` và lấy `caption_text`.

### Phase 6 - Summary validation

File mới đề xuất:

- `audio_pipeline/validation.py`

Hoặc viết function trong `pipeline.py` nếu muốn ít file hơn.

Output mới:

- `features/audio_quality_summary.json`

Nội dung:

```json
{
  "video_id": "L22_V012",
  "num_asr_segments": 226,
  "num_audio_features": 226,
  "num_caption_usable": 214,
  "num_boilerplate_filtered": 8,
  "num_duplicate_filtered": 8,
  "timeline_sorted": true,
  "num_out_of_order": 0,
  "quality_counts": {
    "good": 180,
    "medium": 34,
    "bad": 12,
    "empty": 0
  }
}
```

Acceptance:

- Có summary để audit nhanh sau mỗi run.
- Nếu `timeline_sorted=false`, pipeline nên log warning rõ.

---

## 4. Test plan

### 4.1. Unit-level checks

Test các function:

- `clean_transcript()`
- `normalize_for_quality()`
- `is_boilerplate_transcript()`
- `score_quality()`
- job fingerprint stable/different khi config đổi
- sort helper

### 4.2. Golden sample checks

Dùng output hiện tại của `L22_V012`.

Các assertion:

- `asr_segments.jsonl` sorted theo `start_sec`.
- `audio_features.jsonl` sorted theo `start_sec`.
- Các transcript có `subscribe`, `không bỏ lỡ`, `Amara.org` có:
  - `usable_for_caption=false`
  - `caption_text=""`
  - `reasons` chứa `boilerplate`
- `asr_job_manifest.jsonl` không còn `pending`.
- `video_manifest.jsonl` không còn `pending`.

### 4.3. Resume checks

Chạy pipeline 2 lần cùng config:

- Không duplicate ASR rows.
- Số row giữ nguyên.

Đổi một config ASR nhỏ, ví dụ `beam_size`:

- Fingerprint đổi.
- Pipeline không skip nhầm job cũ.

---

## 5. Thứ tự triển khai khuyến nghị

1. Sửa status manifest.
2. Thêm sort timeline cho ASR/quality/features.
3. Thêm boilerplate + duplicate transcript quality gate.
4. Thêm caption-ready fields.
5. Thêm job fingerprint resume.
6. Thêm validation summary.
7. Chạy lại smoke test `L22_V012`.

---

## 6. Definition of Done

Audio pipeline được coi là sửa xong khi:

- Không còn `pending` gây hiểu nhầm trong manifest output đã hoàn tất.
- `asr_segments.jsonl`, `asr_quality_report.jsonl`, `audio_features.jsonl` đều theo timeline.
- Boilerplate/outro/subscribe không còn `usable_for_caption=true`.
- `audio_features.jsonl` có `caption_text`, `search_text`, `time_window`, `exclude_from_caption_reason`.
- Resume không skip nhầm khi config thay đổi.
- Có summary validation để biết nhanh output có sạch và sorted không.

