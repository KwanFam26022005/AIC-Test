# OCR VLM Pipeline V2 Code Fix Plan

## 0. Mục tiêu

Sửa `ocr_vlm_pipeline_v2` để output OCR dùng được ổn định cho caption fusion, Elasticsearch search và alignment với object/caption/audio.

Phạm vi chính:

- Chuẩn hóa schema frame-level giống object/caption.
- Bổ sung timestamp từ keyframe map CSV.
- Sửa JSONL ordering khi chạy multi-worker.
- Sửa `ocr_text_search` đang bị lặp text và kiểm soát text cần review.
- Thêm validation summary để biết output đã đủ dùng chưa.
- Thêm adapter enrich output cũ để không bắt buộc chạy lại OCR/VLM ngay.

Không thay đổi mục tiêu nhận dạng OCR cốt lõi trong plan này: PaddleOCR detector, VietOCR, Vintern fallback, grouping, scoring/gating vẫn giữ kiến trúc hiện tại.

---

## 1. Hiện trạng từ output cũ

Output kiểm tra:

```text
D:\Users\outputs\ocr_vlm_pipeline_v2\L22_V012
├── L22_V012_ocr_es_docs.jsonl
├── L22_V012_ocr_frame_summary.csv
└── L22_V012_ocr_timing_summary.json
```

Kết quả đo trên `L22_V012_ocr_es_docs.jsonl`:

```text
total docs                         = 283
missing timestamp_sec              = 283
missing canonical_frame_id         = 283
prefixed frame_id                  = 0
ocr_text_search != ocr_text_clean  = 269
frames with ocr_text_review        = 95
review_text directly in search     = 0
need_review && keep_for_index lines= 293
```

Ví dụ record đầu:

```json
{
  "schema_version": "ocr_vlm_pipeline_v2_es_1",
  "document_id": "L22_V012:001",
  "video_id": "L22_V012",
  "frame_id": "001",
  "frame_number": 1,
  "image_path": "/tmp2/.../keyframe_test/L22_V012/001.jpg",
  "ocr_text_clean": "giây",
  "ocr_text_search": "giây giây giây"
}
```

Vấn đề rõ nhất:

- `frame_id` là `001`, trong khi object/caption cần key chung dạng `L22_V012_001`.
- Không có `timestamp_sec`, nên không align được với audio/caption theo thời gian.
- `ocr_text_search` bị lặp do ghép cả `clean_text`, `group_text_clean`, `line_texts`.
- JSONL bị out-of-order khi `workers > 1`.
  - 25 dòng đầu hiện là: `001, 003, 004, 002, 006, 007, ...`
- Line `need_review=True` vẫn có thể `keep_for_index=True`, dễ làm text nghi ngờ lọt vào search chính nếu clean/group builder lấy lại line đó.

---

## 2. Schema đích

Đổi schema:

```text
ocr_vlm_pipeline_v2_es_1 -> ocr_vlm_pipeline_v2_es_2
```

Document frame-level mới:

```json
{
  "schema_version": "ocr_vlm_pipeline_v2_es_2",
  "document_id": "ocr:L22_V012_001",
  "video_id": "L22_V012",
  "frame_id": "L22_V012_001",
  "canonical_frame_id": "L22_V012_001",
  "frame_name": "001",
  "keyframe_idx": 1,
  "source_frame_idx": 0,
  "timestamp_sec": 0.0,
  "timestamp_source": "keyframe_map_csv",
  "image_path": "/tmp2/.../keyframe_test/L22_V012/001.jpg",
  "image_relpath": "L22_V012/001.jpg",
  "ocr_text_clean": "giây",
  "ocr_text_review": "",
  "ocr_text_search": "giây",
  "ocr_text_unaccent": "giay",
  "ocr_terms": ["giay"],
  "quality": {
    "has_clean_text": true,
    "has_review_text": false,
    "num_need_review_lines": 0,
    "num_search_lines": 1
  }
}
```

Giữ backward-compatible fields nếu cần:

- `legacy_frame_id`: `"001"`
- `frame_number`: `1`
- `media.frame_name`
- `media.legacy_frame_id`

Quy ước chung với object/caption:

- `frame_id` và `canonical_frame_id` đều là `video_id + "_" + frame_name`.
- `frame_name` là stem ảnh: `"001"`.
- `keyframe_idx` là số thứ tự keyframe, lấy từ `001.jpg -> 1`.
- `source_frame_idx` là frame index trong video gốc, lấy từ CSV cột `frame_idx`.
- `timestamp_sec` là giây trong video gốc, lấy từ CSV cột `pts_time`.

---

## 3. Sửa timestamp và keyframe map

### File cần sửa

```text
run_frame_folder.py
ocr_pipeline/outputs.py
```

### CLI/config mới

Thêm vào `run_frame_folder.py`:

```bash
--keyframe_map PATH
--video_manifest PATH
--frames_root PATH
--timestamp_strategy map_or_uniform|map_only|uniform|none
```

Ý nghĩa:

- `--keyframe_map`: file CSV hoặc root folder chứa `<video_id>/<video_id>.csv`.
- `--video_manifest`: fallback uniform timestamp nếu không có CSV.
- `--frames_root`: dùng để sinh/resolve `image_relpath`.
- `--timestamp_strategy`: giống object detection.

### Resolver

Thêm helper:

```python
load_keyframe_map_csv(csv_path)
find_keyframe_map_for_video(keyframe_map_path, video_id)
load_video_manifest(manifest_path)
resolve_timestamp(keyframe_idx, keyframe_map, num_keyframes, manifest_doc, strategy)
```

Với `L22_V012.csv`:

```text
001.jpg -> keyframe_idx=1 -> timestamp_sec=0.0     -> source_frame_idx=0
002.jpg -> keyframe_idx=2 -> timestamp_sec=3.0     -> source_frame_idx=90
003.jpg -> keyframe_idx=3 -> timestamp_sec=8.56667 -> source_frame_idx=257
```

Acceptance:

- 283/283 docs có `timestamp_sec`.
- `timestamp_source = "keyframe_map_csv"` khi có CSV.
- Không còn phụ thuộc caption pipeline tự nội suy timestamp.

---

## 4. Sửa frame identity alignment

### Lỗi hiện tại

OCR:

```json
"frame_id": "001",
"document_id": "L22_V012:001"
```

Object/caption cần:

```json
"frame_id": "L22_V012_001"
```

### Sửa

Trong `build_es_document()`:

- Tách `frame_name = Path(image_path).stem`.
- Tạo `canonical_frame_id = f"{video_id}_{frame_name}"`.
- Gán:

```python
legacy_frame_id = frame_name
frame_id = canonical_frame_id
document_id = f"ocr:{canonical_frame_id}"
```

Trong `media` cũng thêm các field này.

Acceptance:

- OCR, object, caption join được trực tiếp bằng `canonical_frame_id`.
- Không còn cần đoán `"001"` thuộc video nào khi merge nhiều video.

---

## 5. Sửa `ocr_text_search`

### Lỗi hiện tại

`outputs.py` đang tạo:

```python
search_parts = [clean_text, *group_texts, *line_texts]
```

Với frame đầu:

```text
ocr_text_clean  = "giây"
ocr_text_search = "giây giây giây"
```

Điều này làm ES score bị lệch vì cùng một OCR evidence được index nhiều lần.

### Policy mới

Mặc định:

```python
ocr_text_search = clean_text
```

Không đưa `group_text_clean` và `line_texts` vào search chính nếu chúng chỉ là evidence tạo ra `clean_text`.

Thêm optional field nếu muốn ranking có trọng số:

```json
"ocr_text_boosted": "giây"
```

Nhưng `ocr_text_boosted` phải được build có kiểm soát:

- Deduplicate theo normalized line.
- Không lặp cùng text quá 1 lần.
- Không lấy `need_review=True` nếu `es_include_review_in_search=false`.

### Review text policy

Mặc định:

- `ocr_text_review` chỉ dùng audit/debug.
- `ocr_text_review` không vào `ocr_text_search`.
- Line có `need_review=True` không được vào `ocr_text_search`.

Nếu cần search cả review:

```bash
--include_review_in_search
```

hoặc config:

```python
es_include_review_in_search = True
```

Khi bật, nên ghi rõ:

```json
"quality": {
  "review_included_in_search": true
}
```

Acceptance:

- Frame `001`: `ocr_text_search = "giây"`, không còn `"giây giây giây"`.
- `num_search_duplicates = 0` trong validation summary.
- `num_need_review_lines_in_primary_search = 0` khi không bật include review.

---

## 6. Sửa JSONL ordering khi multi-worker

### Lỗi hiện tại

`run_frame_folder.py` dùng:

```python
for future in concurrent.futures.as_completed(...):
    append_jsonl(es_jsonl_path, doc)
```

Do đó JSONL ghi theo frame hoàn thành trước, không theo thứ tự frame.

Output hiện tại 25 dòng đầu:

```text
001, 003, 004, 002, 006, 007, 008, 009, 010, 011, 012, 005, ...
```

### Sửa

Có 2 hướng:

#### Hướng A - đơn giản

- Collect `docs` và `rows` trong memory.
- Sort theo `batch_index`.
- Ghi JSONL/CSV ở cuối.

Phù hợp với mỗi video vài trăm tới vài nghìn keyframes.

#### Hướng B - streaming an toàn hơn

- Dùng `pending_docs: dict[int, doc]`.
- Ghi doc khi `next_write_idx` đã sẵn sàng.
- Không cần giữ toàn bộ video trong memory.

Pseudo:

```python
pending_docs[idx] = doc
while next_write_idx in pending_docs:
    append_jsonl(es_jsonl_path, pending_docs.pop(next_write_idx))
    next_write_idx += 1
```

Khuyến nghị dùng hướng B.

Acceptance:

- JSONL line order đúng `001, 002, 003, ...`.
- Summary CSV và JSONL cùng order.
- Multi-worker vẫn giữ tốc độ xử lý.

---

## 7. Resume và fingerprint

### Lỗi hiện tại

`run_frame_folder.py` luôn:

```python
es_jsonl_path.write_text("", encoding="utf-8")
```

Tức là:

- Không resume được.
- Nếu job lỗi giữa chừng, chạy lại từ đầu.
- Không phân biệt output cũ schema v1 với output mới schema v2.

### Sửa

Thêm CLI:

```bash
--resume
--no_resume
--resume_legacy
--no_resume_legacy
```

Thêm:

```python
run_config_hash = hash(schema_version + model/config/search policy)
frame_fingerprint = hash(run_config_hash + image_path.name)
```

Mỗi doc ghi:

```json
"run_config_hash": "...",
"frame_fingerprint": "..."
```

Skip frame chỉ khi:

- `frame_id` đã có.
- `frame_fingerprint` trùng.

Nếu output legacy không có fingerprint:

- `resume_legacy=true`: skip như cũ.
- `resume_legacy=false`: rerun để nâng schema.

Acceptance:

- Chạy cùng config lần 2 không duplicate JSONL.
- Đổi schema/search policy thì fingerprint đổi, không skip nhầm record cũ.

---

## 8. Validation summary

### File cần thêm hoặc tích hợp

```text
validate_ocr_jsonl.py
```

Hoặc tích hợp vào cuối `run_frame_folder.py`.

Output:

```text
<video_id>_ocr_quality_summary.json
```

Nội dung đề xuất:

```json
{
  "schema_version": "ocr_vlm_pipeline_v2_es_2",
  "video_id": "L22_V012",
  "num_frames": 283,
  "num_frames_missing_timestamp": 0,
  "num_frames_missing_canonical_frame_id": 0,
  "num_docs_out_of_order": 0,
  "num_search_duplicates": 0,
  "num_need_review_lines_in_primary_search": 0,
  "num_frames_with_clean_text": 190,
  "num_frames_with_review_text": 95,
  "total_clean_chars": 12345,
  "total_review_chars": 678,
  "top_ocr_terms": {
    "benh": 20,
    "viet": 12
  }
}
```

Acceptance:

- Summary báo rõ output đủ dùng cho caption/search chưa.
- Nếu timestamp/order/search duplicate lỗi, summary phải hiện số khác 0.

---

## 9. Adapter enrich output cũ

Không phải lúc nào cũng cần chạy lại OCR/VLM.

Tạo script:

```text
enrich_ocr_jsonl.py
```

Input:

```bash
python enrich_ocr_jsonl.py \
  --input outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs.jsonl \
  --output outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs_enriched.jsonl \
  --summary-output outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_enriched_summary.json \
  --video-id L22_V012 \
  --keyframe-map keyframe_test
```

Chức năng:

- Đọc output v1.
- Thêm:
  - `schema_version = ocr_vlm_pipeline_v2_es_2`
  - `canonical_frame_id`
  - `frame_id` canonical
  - `legacy_frame_id`
  - `frame_name`
  - `keyframe_idx`
  - `source_frame_idx`
  - `timestamp_sec`
  - `timestamp_source`
  - `image_relpath`
  - `ocr_text_search` dedup/clean-only
  - `ocr_terms`
  - validation summary
- Sort JSONL theo `keyframe_idx`.

Không thể sửa lại OCR recognition đã sai, nhưng đủ để unblock caption/retrieval baseline.

Acceptance:

- Enrich được 283 docs `L22_V012` không cần chạy lại model.
- `num_frames_missing_timestamp = 0`.
- `num_docs_out_of_order = 0`.
- `num_search_duplicates = 0`.

---

## 10. Elasticsearch mapping khuyến nghị

Vì OCR/object/caption đều là text-like signals, nên schema ES nên tách rõ:

### Keyword/filter fields

```json
{
  "video_id": "keyword",
  "frame_id": "keyword",
  "canonical_frame_id": "keyword",
  "frame_name": "keyword",
  "schema_version": "keyword",
  "timestamp_source": "keyword",
  "ocr_terms": "keyword"
}
```

### Numeric fields

```json
{
  "keyframe_idx": "integer",
  "source_frame_idx": "integer",
  "timestamp_sec": "float",
  "quality.clean_chars": "integer",
  "quality.review_chars": "integer",
  "quality.num_keep_lines": "integer",
  "quality.num_review_lines": "integer"
}
```

### Text fields

```json
{
  "ocr_text_search": "text",
  "ocr_text_unaccent": "text",
  "ocr_text_review": "text",
  "all_ocr_text": "text"
}
```

Khuyến nghị:

- `ocr_text_search`: primary search, clean/indexable only.
- `ocr_text_unaccent`: analyzer không dấu hoặc normalized field cho tiếng Việt.
- `ocr_text_review`: không search mặc định; dùng audit hoặc search phụ khi user bật.
- `ocr_terms`: keyword tokens để filter/autocomplete nhẹ.

### Nested evidence

Nếu cần query line-level bbox:

```json
"ocr_lines": {
  "type": "nested",
  "properties": {
    "final_text": {"type": "text"},
    "keep_for_index": {"type": "boolean"},
    "need_review": {"type": "boolean"},
    "bbox_xyxy": {"type": "float"}
  }
}
```

Nếu chỉ search frame-level, có thể keep `ocr_lines` là object thường để giảm mapping complexity.

---

## 11. Test plan

### Unit checks

Test helper:

- `extract_frame_name()`
- `make_canonical_frame_id()`
- `load_keyframe_map_csv()`
- `resolve_timestamp()`
- `build_ocr_search_text()`
- `dedupe_search_parts()`
- `is_jsonl_ordered()`

### Golden checks với `L22_V012`

Assertions:

- 283 docs.
- 283 docs có `timestamp_sec`.
- 283 docs có `canonical_frame_id`.
- `frame_id` dạng `L22_V012_001`.
- JSONL đúng thứ tự `001 -> 283`.
- Frame `001`: `ocr_text_search` không lặp `"giây giây giây"`.
- `num_need_review_lines_in_primary_search = 0` khi `include_review=false`.
- `image_relpath = L22_V012/001.jpg`.

### Resume checks

- Chạy cùng config 2 lần với `--resume`: không duplicate.
- Đổi `schema_version` hoặc search policy: fingerprint đổi.
- Output legacy v1 có thể rerun khi `--no_resume_legacy`.

---

## 12. Thứ tự triển khai khuyến nghị

1. Thêm helper metadata/timestamp/keyframe map.
2. Sửa `build_es_document()` sang schema v2.
3. Sửa `ocr_text_search` clean-only + dedup policy.
4. Sửa multi-worker JSONL ordering.
5. Thêm validation summary.
6. Thêm resume/fingerprint.
7. Thêm adapter `enrich_ocr_jsonl.py` cho output cũ.
8. Cập nhật README/lệnh server.
9. Chạy golden check trên `L22_V012`.

---

## 13. Definition of Done

OCR pipeline được coi là sửa xong khi:

- Mỗi record có `frame_id`, `canonical_frame_id`, `frame_name`, `keyframe_idx`, `timestamp_sec`.
- `frame_id` join được trực tiếp với object/caption.
- `ocr_text_search` không lặp line/group/clean vô kiểm soát.
- Text `need_review=True` không vào search chính khi chưa bật option.
- JSONL output ordered theo keyframe.
- Có `image_relpath`.
- Có validation summary báo lỗi schema/search/timestamp/order.
- Có adapter enrich output cũ để unblock caption baseline.
