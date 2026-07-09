# Object Detection Code Fix Plan

Mục tiêu của plan này là sửa `data_processing/object_detection/` để output RAM++ + GroundingDINO trở thành nguồn evidence ổn định cho caption fusion, temporal retrieval và search indexing.

Scope chính:

- Không thay model RAM++ / GroundingDINO ở bước đầu.
- Ưu tiên sửa schema, timestamp alignment, object metadata, count deduplication và validation.
- Giữ backward compatibility với output hiện tại (`ram_gdino_object_detection_v1`) nếu có thể.
- Caption pipeline nên dùng các field đã chuẩn hóa mới, không phải tự đoán từ `objects` thô.

---

## 1. Hiện trạng lỗi cần sửa

### 1.1. Thiếu timestamp cho mọi frame

Output thực tế:

- `outputs/object_detection/L22_V012_objects.jsonl`
- 283 / 283 records không có `timestamp_sec`.
- Hiện đã có map keyframe mới: `keyframe_test/L22_V012/L22_V012.csv`.

Ảnh hưởng:

- Không align được audio window theo frame.
- Không group shot bằng timestamp gap.
- Không build TRAKE/event-step theo thứ tự thời gian một cách chắc chắn.

Giải thích timestamp trong bài toán này:

- `timestamp_sec` không phải thời gian thật ngoài đời.
- `timestamp_sec` là mốc thời gian của keyframe trong video gốc, tính bằng giây từ đầu video.
- Với frame đã extract như `001.jpg`, timestamp chính xác nên lấy từ CSV map keyframe.
- CSV hiện có các cột:
  - `n`: thứ tự keyframe, tương ứng tên ảnh `001.jpg`, `002.jpg`, ...
  - `pts_time`: timestamp giây trong video gốc.
  - `fps`: FPS video.
  - `frame_idx`: frame index trong video gốc.

Ví dụ:

```csv
n,pts_time,fps,frame_idx
1,0.0,30.0,0
2,3.0,30.0,90
3,8.56667,30.0,257
```

Mapping đúng:

```text
001.jpg -> keyframe_idx=1 -> timestamp_sec=0.0 -> source_frame_idx=0
002.jpg -> keyframe_idx=2 -> timestamp_sec=3.0 -> source_frame_idx=90
003.jpg -> keyframe_idx=3 -> timestamp_sec=8.56667 -> source_frame_idx=257
```

Code liên quan:

- `ram_gdino_pipeline.py`
- `build_frame_doc()`

### 1.2. `frame_idx` hiện đang nhập nhằng nghĩa

Hiện tại:

```json
{
  "frame_id": "L22_V012_001",
  "frame_idx": 1,
  "frame_name": "001"
}
```

Vấn đề:

- `frame_idx` được lấy từ tên file keyframe `001.jpg`.
- Trong CSV map keyframe, `frame_idx` lại là original video frame number.
- Dùng chung tên `frame_idx` cho hai ý nghĩa sẽ gây lỗi timestamp và alignment.

Giải pháp:

- Thêm `keyframe_idx`.
- Thêm `source_frame_idx` lấy từ cột `frame_idx` của CSV.
- Thêm `timestamp_sec` lấy từ cột `pts_time` của CSV.
- Giữ `frame_idx` tạm thời để backward compatibility, nhưng document rõ nó là keyframe ordinal trong output v1.

### 1.3. Thiếu object metadata cho caption

Output hiện tại thiếu cho toàn bộ 1586 objects:

- `area_ratio`
- `position`
- `bbox_xyxy` alias
- `confidence` alias
- `label_lower`
- `size_bucket`

Ảnh hưởng:

- Caption fuser không biết object nào lớn/quan trọng.
- Không mô tả được vị trí như “bên trái”, “giữa ảnh”, “phía dưới”.
- Downstream phải tự parse `box`, dễ lặp logic và sai schema.

Code liên quan:

- `tag_canonicalization.py`
- `ram_gdino_pipeline.py`

### 1.4. Object counts bị phồng do hierarchy collision

Ví dụ thực tế:

```json
{
  "frame_id": "L22_V012_003",
  "object_counts": {
    "PERSON": 2,
    "SCREEN": 1,
    "WOMAN": 1
  }
}
```

Trong frame này, `WOMAN` overlap với một `PERSON`, nên caption có thể hiểu sai thành 3 người.

Thống kê hiện tại:

- 22 frames có collision kiểu `PERSON` + `MAN/WOMAN/STUDENT/CHILD_STUDENT`.

Code liên quan:

- `tag_canonicalization.py`
- `canonicalize_detections()`
- `class_agnostic_nms()`

### 1.5. Scene labels vẫn lọt vào object counts

Ví dụ:

```json
{
  "frame_id": "L22_V012_001",
  "object_counts": {
    "SKY": 1,
    "CITY": 1,
    "WATER": 1,
    "SUN": 2
  }
}
```

Vấn đề:

- `SKY`, `CITY`, `WATER`, `SUN`, `SEA`, `NIGHT` nên là `scene_tags` hoặc `image_tags`, không nên là object counts chính.
- Scene threshold theo area ratio chưa đủ vì scene-like label có thể có bbox nhỏ hơn threshold.

Thống kê hiện tại:

- 33 frames còn scene-like labels trong final objects/counts.

Code liên quan:

- `tag_canonicalization.py`
- `NON_OBJECT_TAGS`
- `normalize_and_filter()`

### 1.6. Image path: giữ server path là canonical, thêm relpath chỉ để tiện fallback

Output hiện tại:

```json
"image_path": "/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test/L22_V012/001.jpg"
```

Review cập nhật:

- Pipeline thực tế sẽ chạy trên server.
- Vì vậy `image_path` absolute trên server là hợp lệ và nên giữ làm canonical runtime path.
- Việc path không tồn tại trên máy local không phải blocker nếu code được push GitHub rồi server `git pull`.

Rủi ro còn lại:

- Nếu export ES hoặc debug ở môi trường khác, absolute server path không portable.
- Nếu caption/VLM chạy cùng server thì không vấn đề.

Giải pháp:

- Giữ `image_path` là field chính.
- Thêm `image_relpath`: `L22_V012/001.jpg` như optional fallback/debug field.
- Thêm `frame_name`: `001`.
- Downstream trên server ưu tiên dùng `image_path`.
- Tool local/debug có thể dùng `frames_root + image_relpath`.

### 1.7. Label schema cần tối ưu cho ES/search/caption

Hiện tại:

- `object_counts`: uppercase (`PERSON`, `WINDOW`, `WOMAN`)
- `tags`: lowercase (`person`, `news`, `screen`)

Vấn đề:

- Caption/search cần lowercase normalized text field.
- ES filter cần keyword field ổn định.
- Query dạng “có ít nhất 2 người”, “người và xe máy”, “cảnh thành phố buổi tối” cần field riêng cho countable object, scene tags và RAM image tags.
- Không nên chỉ lưu một field `objects` nested rồi bắt search/caption tự xử lý.

Giải pháp:

- Giữ `object_counts` uppercase nếu cần backward compatibility.
- Thêm:
  - `object_counts_normalized`
  - `object_count_items`
  - `object_tags`
  - `scene_tags`
  - `ram_tags`
  - `object_text`
  - `scene_text`
  - `ram_tag_text`
  - `all_object_text`
  - `important_objects`

---

## 2. Schema output đề xuất

Tăng schema version:

```text
ram_gdino_object_detection_v1_1
```

Ví dụ record sau sửa:

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
  "image_path": "/tmp2/.../L22_V012/001.jpg",
  "image_relpath": "L22_V012/001.jpg",
  "image_size": [1280, 720],
  "tags": ["boat", "city", "sunset", "water"],
  "ram_tags": ["boat", "city", "sunset", "water"],
  "scene_tags": ["city", "sky", "sunset", "water"],
  "object_tags": ["person", "screen", "car"],
  "object_counts": {
    "PERSON": 2,
    "SCREEN": 1
  },
  "object_counts_normalized": {
    "person": 2,
    "screen": 1
  },
  "object_count_items": [
    {"label": "person", "count": 2},
    {"label": "screen", "count": 1}
  ],
  "important_objects": [
    "2 persons",
    "1 screen"
  ],
  "object_text": "person person screen",
  "scene_text": "city sky sunset water",
  "ram_tag_text": "boat city sunset water",
  "all_object_text": "person person screen city sky sunset water boat city sunset water",
  "objects": [
    {
      "label": "PERSON",
      "label_lower": "person",
      "raw_label": "WOMAN",
      "score": 0.4928,
      "confidence": 0.4928,
      "box": [752.55, 200.49, 868.63, 603.17],
      "bbox_xyxy": [752.55, 200.49, 868.63, 603.17],
      "area_ratio": 0.051,
      "position": "right",
      "center_xy": [810.59, 401.83],
      "size_bucket": "medium",
      "source": "groundingdino",
      "countable": true
    }
  ],
  "quality": {
    "num_raw_tags": 11,
    "num_prompt_tags": 7,
    "num_raw_boxes": 4,
    "num_final_boxes": 3,
    "num_countable_objects": 2,
    "num_scene_labels": 1,
    "num_deduped_hierarchy": 1,
    "num_non_countable": 1
  }
}
```

### 2.1. ES/search mapping strategy

Vì object extraction, OCR và caption đều là text-heavy evidence, schema nên tách rõ field cho search thay vì dồn vào một blob.

Khuyến nghị ES fields:

```yaml
keyword_fields:
  video_id: keyword
  frame_id: keyword
  canonical_frame_id: keyword
  object_tags: keyword
  scene_tags: keyword
  ram_tags: keyword

numeric_fields:
  keyframe_idx: integer
  source_frame_idx: integer
  timestamp_sec: float

text_fields:
  object_text: simple_or_english_analyzer
  scene_text: simple_or_english_analyzer
  ram_tag_text: simple_or_english_analyzer
  all_object_text: simple_or_english_analyzer

count_fields:
  object_count_items: nested_or_flattened
```

Ghi chú:

- `object_tags`, `scene_tags`, `ram_tags` dùng cho filter/exact matching.
- `object_text` nên lặp label theo count, ví dụ `person person screen`, để lexical search ưu tiên frame có nhiều object liên quan.
- `object_count_items` tốt hơn dynamic field kiểu `object_counts_normalized.person` nếu label open-vocabulary quá rộng.
- `objects` chi tiết bbox nên giữ để UI/debug/caption evidence, nhưng không nhất thiết dùng làm field search chính.
- Khi build compact search index sau này, `all_object_text` sẽ được fuse với OCR/audio/caption.

---

## 3. Code changes theo phase

### Phase 1 - Metadata alignment và timestamp

File: `ram_gdino_pipeline.py`

Thêm CLI/config:

```yaml
metadata:
  video_manifest: null
  keyframe_map: null
  timestamp_strategy: "map_or_uniform"
  frames_root: null
```

CLI tương ứng:

```text
--video-manifest
--keyframe-map
--frames-root
--timestamp-strategy map_or_uniform|map_only|uniform|none
```

Việc cần làm:

- Load keyframe map CSV nếu có. Với sample hiện tại:

```text
keyframe_test/L22_V012/L22_V012.csv
```

- Map theo cột `n` với tên frame:

```python
keyframe_idx = int(frame_name)       # "001" -> 1
map_row = keyframe_map[keyframe_idx]
timestamp_sec = float(map_row["pts_time"])
source_frame_idx = int(map_row["frame_idx"])
timestamp_source = "keyframe_map_csv"
```

- Load `video_manifest.jsonl` nếu cần fallback `duration_sec`, `fps`.
- Nếu không có map, fallback:

```python
timestamp_sec = (keyframe_idx - 1) * duration_sec / max(1, num_keyframes - 1)
timestamp_source = "uniform_interpolation"
```

- Nếu có cả `pts_time` và `frame_idx/fps`, ưu tiên `pts_time` vì đây là timestamp đã được extractor ghi ra.

- Thêm field:
  - `keyframe_idx`
  - `source_frame_idx`
  - `timestamp_sec`
  - `timestamp_source`

Acceptance:

- Với `L22_V012.csv`, 100% object records có `timestamp_sec` từ map CSV.
- Không còn phụ thuộc vào caption pipeline để tự nội suy timestamp.

### Phase 2 - Image path server-first, relpath optional

File: `ram_gdino_pipeline.py`

Việc cần làm:

- Trong `build_frame_doc()`, thêm:

```python
image_relpath = f"{video_id}/{image_path.name}"
```

- Nếu input đang là parent root và batch mode, vẫn giữ relpath theo `video_id/frame_name.ext`.
- Giữ `image_path` absolute làm path chính trên server.
- Downstream resolver ưu tiên:
  1. `image_path` nếu tồn tại
  2. `frames_root / image_relpath`
  3. `frames_dir / frame_name`

File: `visualize_detections.py`

Việc cần làm:

- Cập nhật `resolve_image_path()` để dùng `image_relpath`.

Acceptance:

- Trên server, `image_path` vẫn resolve trực tiếp.
- Local/debug có thể resolve bằng `--frames-root .../keyframe_test` nếu cần.

### Phase 3 - Object geometry enrichment

File: `tag_canonicalization.py`

Thêm helper:

```python
def compute_position(bbox, img_width, img_height) -> str:
    ...

def size_bucket(area_ratio: float) -> str:
    ...

def enrich_detection(det, img_width, img_height) -> dict:
    ...
```

Logic đề xuất:

- `position` dựa trên center:
  - `left`, `center`, `right`
  - có thể thêm `top_left`, `bottom_right` nếu cần chi tiết hơn.
- `size_bucket`:
  - `< 0.01`: `small`
  - `< 0.08`: `medium`
  - else: `large`

Mỗi object thêm:

- `label_lower`
- `raw_label`
- `confidence`
- `bbox_xyxy`
- `area_ratio`
- `center_xy`
- `position`
- `size_bucket`
- `countable`

Acceptance:

- 100% objects có `area_ratio`, `position`, `bbox_xyxy`, `confidence`.

### Phase 4 - Scene và non-countable labels

File: `tag_canonicalization.py`

Thêm cấu hình label sets:

```python
SCENE_LABELS = {
    "SKY", "CITY", "CITY_SKYLINE", "WATER", "SEA", "SUN", "SUNSET",
    "NIGHT", "ROAD", "LANDSCAPE", "BACKGROUND"
}

NON_COUNTABLE_LABELS = {
    "HAND", "ARM", "LEG", "FACE", "HEAD",
    "TIE", "UNIFORM", "DRESS_SHIRT"
}
```

Rule:

- `SCENE_LABELS` chuyển sang `scene_labels` / `scene_tags`.
- `NON_COUNTABLE_LABELS` giữ trong `objects` nếu cần evidence, nhưng:
  - `countable=false`
  - không đưa vào `object_counts_normalized`
  - có thể đưa vào `object_tags` nếu hữu ích.

Acceptance:

- `SKY/CITY/WATER/SUN` không còn phồng `object_counts_normalized`.
- Caption evidence phân biệt rõ scene tags và countable objects.

### Phase 5 - Hierarchy/person-family deduplication

File: `tag_canonicalization.py`

Thêm family map:

```python
COUNT_CANONICAL_MAP = {
    "MAN": "PERSON",
    "WOMAN": "PERSON",
    "BOY": "PERSON",
    "GIRL": "PERSON",
    "STUDENT": "PERSON",
    "CHILD_STUDENT": "PERSON"
}
```

Thiết kế:

- Giữ `raw_label` để biết model ban đầu detect gì.
- Dùng `label` canonical cho count/search.
- Với box overlap cùng family:
  - nếu IoU cao hoặc box containment cao, giữ detection confidence tốt hơn.
  - merge raw labels vào `attributes` nếu cần:

```json
"raw_label": "WOMAN",
"label": "PERSON",
"attributes": ["woman"]
```

Thêm helper:

```python
def canonical_count_label(label: str) -> str:
    ...

def family_aware_nms(detections, iou_threshold, containment_threshold):
    ...
```

Acceptance:

- Frame `L22_V012_003` không còn count `PERSON=2` + `WOMAN=1`; count normalized nên là `person: 2`, preserve `woman` trong attributes/tags nếu muốn.
- Các frame có `PERSON` + specific person label không bị over-count.

### Phase 6 - Caption/search-ready fields

File: `ram_gdino_pipeline.py`

Thêm output builder:

```python
def build_object_counts(objects):
    ...

def build_object_text(object_counts_normalized):
    ...

def build_important_objects(object_counts_normalized):
    ...
```

Field thêm:

- `object_tags`
- `scene_tags`
- `object_counts_normalized`
- `object_text`
- `important_objects`

Rule:

- `object_counts` giữ dạng legacy uppercase.
- Caption pipeline dùng `object_counts_normalized`, `important_objects`, `scene_tags`.
- Search index dùng `object_text`.

Acceptance:

- Caption evidence không phải tự lowercase/count.
- Search có field object text ổn định.

### Phase 7 - Resume safety bằng config fingerprint

File: `ram_gdino_pipeline.py`

Vấn đề hiện tại:

- `--resume` skip chỉ theo `frame_id`.
- Nếu đổi threshold, model, tag filter hoặc schema, output cũ vẫn bị skip.

Thêm:

- `run_config_hash`
- `frame_fingerprint`

Fingerprint nên gồm:

```text
schema_version
ram checkpoint/model
ram image size
gdino model id
image max side
box threshold
text threshold
nms threshold
scene threshold
canonicalization version
input image path/name
```

Resume rule:

- Chỉ skip khi `frame_id` và `frame_fingerprint` khớp.
- Nếu output cũ không có fingerprint, warning và tùy config:
  - `resume_legacy=true`: skip như cũ
  - `resume_legacy=false`: rerun

Acceptance:

- Đổi threshold không dùng nhầm output cũ.

### Phase 8 - Validation summary

File mới đề xuất:

- `object_detection_validation.py`

Hoặc tích hợp vào cuối `ram_gdino_pipeline.py`.

Output mới:

- `outputs/object_detection/summaries/<video_id>_quality_summary.json`

Nội dung:

```json
{
  "video_id": "L22_V012",
  "num_frames": 283,
  "num_objects": 1586,
  "num_frames_missing_timestamp": 0,
  "num_objects_missing_area_ratio": 0,
  "num_objects_missing_position": 0,
  "num_person_family_collisions": 0,
  "num_scene_labels_in_counts": 0,
  "num_unresolved_image_paths": 0,
  "top_object_counts": {
    "person": 272,
    "window": 118
  },
  "schema_version": "ram_gdino_object_detection_v1_1"
}
```

Acceptance:

- Có summary để biết nhanh output đủ dùng cho caption chưa.
- Nếu còn missing timestamp/object geometry, summary phải báo rõ.

---

## 4. Adapter/migration cho output cũ

Không phải lỗi nào cũng cần chạy lại GPU.

Tạo script:

```text
enrich_object_detection_jsonl.py
```

Chức năng:

- Đọc output v1 hiện có.
- Thêm:
  - `keyframe_idx`
  - `timestamp_sec` fallback từ manifest
  - `image_relpath`
  - `area_ratio`
  - `position`
  - `bbox_xyxy`
  - `confidence`
  - `object_counts_normalized`
  - `object_text`
  - `important_objects`
- Không thể sửa hoàn hảo raw detection nếu đã bị NMS mất thông tin, nhưng đủ để unblock caption baseline.

Input:

```bash
python enrich_object_detection_jsonl.py \
  --input outputs/object_detection/L22_V012_objects.jsonl \
  --output outputs/object_detection/L22_V012_objects_enriched.jsonl \
  --video-manifest outputs/audio/L22_V012/manifests/video_manifest.jsonl \
  --frames-root keyframe_test
```

Acceptance:

- Có thể tạo enriched JSONL cho `L22_V012` mà không cần chạy lại RAM/GDINO.

---

## 5. Test plan

### 5.1. Unit checks

Test các helper:

- `compute_area_ratio()`
- `compute_position()`
- `size_bucket()`
- `canonical_count_label()`
- `family_aware_nms()`
- scene/non-countable label filtering
- timestamp resolver từ map CSV và uniform fallback

### 5.2. Golden checks với `L22_V012`

Assertions:

- 283 records.
- 283 records có `timestamp_sec`.
- 1586 objects có `area_ratio`, `position`, `bbox_xyxy`, `confidence`.
- Không còn scene labels trong `object_counts_normalized`.
- Frame `L22_V012_003` có `person: 2`, không bị `person + woman` count thành 3.
- `image_relpath` resolve được với local `keyframe_test`.

### 5.3. Resume checks

Chạy cùng config hai lần:

- Không duplicate records.
- Skip đúng frame fingerprint.

Đổi `box_threshold` hoặc `scene_area_threshold`:

- Fingerprint đổi.
- Pipeline không skip nhầm output cũ.

---

## 6. Thứ tự triển khai khuyến nghị

1. Thêm geometry enrichment (`area_ratio`, `position`, `bbox_xyxy`, `confidence`).
2. Thêm image portable path (`image_relpath`).
3. Thêm timestamp resolver (`video_manifest`, `keyframe_map`, uniform fallback).
4. Tách scene/non-countable labels khỏi countable object counts.
5. Sửa person-family hierarchy count.
6. Thêm caption/search-ready fields.
7. Thêm resume fingerprint.
8. Thêm validation summary.
9. Viết adapter enrich output cũ để unblock caption baseline.

---

## 7. Definition of Done

Object detection pipeline được coi là sửa xong khi:

- Mỗi frame có `canonical_frame_id`, `keyframe_idx`, `timestamp_sec`, `timestamp_source`.
- Mỗi object có `bbox_xyxy`, `confidence`, `area_ratio`, `position`, `size_bucket`.
- Có `image_relpath` portable.
- `object_counts_normalized` không đếm scene labels.
- Person-family labels không làm phồng số người.
- Có `scene_tags`, `object_tags`, `object_text`, `important_objects`.
- Resume không skip nhầm khi config thay đổi.
- Có validation summary báo rõ output đã đủ dùng cho caption/retrieval.
