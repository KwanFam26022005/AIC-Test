# Kế hoạch tái cấu trúc pipeline PP-OCRv6 + VLM Correction + Elasticsearch

**Mục tiêu vòng này:** chuyển notebook demo hiện tại thành pipeline OCR + VLM correction có thể chạy thực tế trên khoảng **1.000.000 frames** với server **1 GPU RTX A5000**, có khả năng lưu OCR text vào Elasticsearch và hỗ trợ fuzzy search.

**Ngoài scope vòng này:** object detection và color extraction sẽ được xử lý bằng pipeline riêng, sau đó có thể merge vào Elasticsearch ở bước tích hợp sau.

**Input hiện tại:** thư mục frames `.jpg` theo từng video, ví dụ `L25_V001/008.jpg`.

**Output mục tiêu:**

```text
outputs/
  <video_id>/
    frame_registry.parquet
    ppocr_raw.parquet
    ocr_risk.parquet
    ocr_groups.parquet
    vlm_jobs.parquet
    vlm_corrected.parquet
    es_documents.jsonl
    logs/
      runtime_stats.json
      failed_jobs.jsonl
      sample_review.csv
```

---

## 1. Tóm tắt workflow notebook hiện tại

Notebook hiện tại đang làm đúng hướng cho demo:

```text
Frames JPG
  ↓
PP-OCRv6 detect + recognize
  ↓
Lưu OCR raw ra parquet
  ↓
Confidence gating với CONF_THRESHOLD = 0.75
  ↓
Load Vintern-3B-R-beta 4-bit
  ↓
Group bbox theo proximity
  ↓
Crop group region
  ↓
Vintern correction
  ↓
Parse output và merge corrected text
```

Baseline quan sát từ notebook demo:

```text
Số frames demo: 438
Số OCR text lines: 5.109
Thời gian PP-OCRv6: khoảng 260 giây
Tốc độ PP-OCRv6: khoảng 0,59 giây/frame trên môi trường Colab demo
Low-confidence theo threshold 0.75: khoảng 4,4% OCR lines
```

Nhận xét chính:

1. Không nên gọi VLM cho toàn bộ frame khi scale lên 1 triệu frame.
2. Không nên chỉ dựa vào `confidence < 0.75`, vì OCR tiếng Việt có thể sai dấu/chữ dù confidence rất cao.
3. Cần thêm `risk_score`, dedup/cache và pipeline restartable.
4. Cần chuyển notebook thành các module độc lập để agent dễ coding/test.

---

## 2. Kiến trúc mục tiêu

Pipeline production nên là:

```text
Frame registry
  ↓
Frame/ROI dedup
  ↓
PP-OCRv6 pass
  ↓
OCR risk scoring
  ↓
BBox grouping + layout classification
  ↓
VLM job builder
  ↓
VLM cache lookup
  ↓
Selective VLM correction
  ↓
Merge raw + corrected OCR
  ↓
Build Elasticsearch documents
  ↓
Bulk index to Elasticsearch
  ↓
Search API / fuzzy query
```

Nguyên tắc quan trọng:

```text
PP-OCRv6 là engine chính.
VLM chỉ là correction layer.
VLM chỉ chạy trên group có rủi ro lỗi hoặc group chưa từng được sửa.
Raw OCR phải luôn được lưu lại để debug/evaluate.
Corrected OCR là field chính phục vụ search.
Normalized OCR là field phụ để hỗ trợ search không dấu/fuzzy.
```

---

## 3. Chiến lược chọn model Vintern

Pipeline phải hỗ trợ switch model bằng config, không hard-code trong notebook.

### 3.1. Model options

```yaml
vlm:
  enabled: true
  provider: "huggingface_transformers"
  model_id: "5CD-AI/Vintern-3B-beta"       # option A
  # model_id: "5CD-AI/Vintern-3B-R-beta"   # option B
  quantization: "4bit_nf4"
  device: "cuda:0"
  max_new_tokens_default: 128
  do_sample: false
  temperature: 0.0
  output_format: "strict_json"
```

### 3.2. Cho user test

Agent cần code để user có thể chạy benchmark 2 model:

```bash
python run_vlm_benchmark.py \
  --video_id L25_V001 \
  --sample_frames 100 \
  --models 5CD-AI/Vintern-3B-beta 5CD-AI/Vintern-3B-R-beta erax-ai/EraX-VL-2B-V1.5 \
  --output outputs/L25_V001/model_benchmark.csv
```

Benchmark cần lưu:

```text
model_id
num_groups
avg_latency_sec_per_group
p50_latency
p95_latency
gpu_memory_peak_gb
parse_success_rate
json_valid_rate
text_change_rate
manual_accuracy_score nếu có label hoặc review thủ công
```

### 3.3. Lựa chọn đề xuất khi chạy thực tế

Đề xuất default thực tế:

```text
Default production: 5CD-AI/Vintern-3B-beta
Optional fallback: 5CD-AI/Vintern-3B-R-beta
```

Lý do:

1. Nhiệm vụ hiện tại là **OCR correction**, không phải reasoning phức tạp.
2. `Vintern-3B-beta` phù hợp hơn làm model correction tổng quát và output ngắn.
3. `Vintern-3B-R-beta` là reasoning model, nên có thể tạo reasoning/chain output dài hơn, latency cao hơn và parse khó hơn nếu không ép strict JSON.
4. Chỉ dùng `Vintern-3B-R-beta` khi group text thuộc dạng layout phức tạp: bảng biểu, văn bản nhiều dòng, crop mờ, text bị che khuất, hoặc cần suy luận từ ngữ cảnh ảnh.
5. Quyết định cuối cùng phải dựa trên benchmark nội bộ: nếu `Vintern-3B-R-beta` tăng accuracy rõ rệt mà chi phí latency chấp nhận được thì dùng cho nhóm khó; còn lại dùng `Vintern-3B-beta`.

Quy tắc router đề xuất:

```python
if region_type in ["table", "document_block", "complex_signboard"] or risk_score >= 0.85:
    model_id = "5CD-AI/Vintern-3B-R-beta"
else:
    model_id = "5CD-AI/Vintern-3B-beta"
```

Trong giai đoạn đầu để đơn giản:

```text
Chạy benchmark cả 2 model trên cùng sample.
Nếu accuracy gần tương đương: chọn Vintern-3B-beta.
Nếu R-beta tốt hơn rõ ở nhóm khó: dùng hybrid routing.
```

---

## 4. Project structure đề xuất

```text
ocr_vlm_pipeline/
  configs/
    default.yaml
    colab_demo.yaml
    server_a5000.yaml
    es_mapping.json

  src/
    __init__.py
    config.py
    logging_utils.py

    frame_registry.py
    dedup.py

    ppocr_engine.py
    ppocr_runner.py

    risk_scoring.py
    grouping.py
    cropper.py
    layout_classifier.py

    vlm_loader.py
    vlm_prompt.py
    vlm_corrector.py
    vlm_parser.py
    vlm_cache.py
    vlm_router.py

    normalize_text.py
    merge_results.py

    es_schema.py
    es_document_builder.py
    es_bulk_indexer.py
    es_search_examples.py

    metrics.py
    benchmark.py
    validation.py

  scripts/
    01_build_frame_registry.py
    02_run_ppocr.py
    03_build_vlm_jobs.py
    04_run_vlm_correction.py
    05_merge_features.py
    06_build_es_documents.py
    07_bulk_index_es.py
    08_search_demo.py
    run_full_pipeline.py
    run_vlm_benchmark.py

  notebooks/
    demo_colab_refactored.ipynb

  tests/
    test_normalize_text.py
    test_risk_scoring.py
    test_grouping.py
    test_vlm_parser.py
    test_es_document_builder.py
```

---

## 5. Config chuẩn cho server A5000

File: `configs/server_a5000.yaml`

```yaml
project:
  video_id: "L25_V001"
  frames_dir: "/data/frames/L25_V001"
  output_dir: "/data/outputs/L25_V001"
  resume: true

runtime:
  device: "cuda:0"
  seed: 42
  num_workers_io: 8
  save_every_n_frames: 1000
  log_every_n_frames: 100

ppocr:
  enabled: true
  ocr_version: "PP-OCRv6"
  lang: "vi"
  use_doc_orientation_classify: false
  use_doc_unwarping: false
  use_textline_orientation: false
  text_det_limit_side_len: 1216
  text_det_thresh: 0.35
  text_det_box_thresh: 0.60
  text_det_unclip_ratio: 2.0
  text_rec_score_thresh: 0.0

risk:
  confidence_low: 0.75
  confidence_medium: 0.90
  min_text_len: 2
  suspicious_chars: ["Ä", "Ë", "�", "□", "训", "|", "¦"]
  enable_vietnamese_spell_rules: true
  enable_temporal_disagreement: true
  vlm_risk_threshold: 0.45

grouping:
  vertical_gap: 15
  horizontal_gap: 15
  crop_padding: 20
  min_crop_side: 64
  max_crop_side: 1024

vlm:
  enabled: true
  model_id: "5CD-AI/Vintern-3B-beta"
  alternative_model_id: "5CD-AI/Vintern-3B-R-beta"
  allow_model_switch: true
  quantization: "4bit_nf4"
  do_sample: false
  temperature: 0.0
  max_new_tokens_base: 64
  max_new_tokens_per_line: 24
  max_new_tokens_cap: 192
  batch_size: 1
  strict_json: true
  cache_enabled: true
  cache_path: "/data/outputs/L25_V001/vlm_cache.sqlite"

es:
  enabled: true
  hosts: ["http://localhost:9200"]
  index_name: "video_frames_v1"
  bulk_size: 500
  request_timeout: 120
```

---

## 6. Module specs chi tiết cho agent

### 6.1. `frame_registry.py`

Nhiệm vụ:

```text
Quét folder frames.
Tạo dataframe frame-level metadata.
Lưu frame_id, frame_number, frame_path, width, height, file_size, phash nếu có.
```

Schema output `frame_registry.parquet`:

```text
video_id: string
frame_id: string
frame_number: int
frame_path: string
width: int
height: int
file_size: int
phash: string|null
scene_id: string|null
is_duplicate: bool
source_status: string
```

Acceptance:

```text
Không crash nếu thiếu/corrupt frame.
Frame corrupt ghi source_status="error".
Có resume mode.
```

---

### 6.2. `dedup.py`

Nhiệm vụ:

```text
Dedup frame trước OCR nếu frame gần giống nhau.
Dedup crop trước VLM nếu cùng text/crop/layout đã correction.
```

Dedup levels:

```text
Level 1: frame-level phash.
Level 2: region/crop-level phash.
Level 3: VLM job-level cache key.
```

Cache key VLM:

```python
cache_key = sha1(
    normalize_for_cache(raw_group_text) + "|" +
    crop_phash + "|" +
    bbox_layout_signature + "|" +
    model_id + "|" +
    prompt_version
).hexdigest()
```

Acceptance:

```text
Nếu cache hit thì không gọi VLM.
Cache phải lưu raw_text, corrected_text, model_id, prompt_version, created_at.
```

---

### 6.3. `ppocr_engine.py` và `ppocr_runner.py`

Nhiệm vụ:

```text
Khởi tạo PP-OCRv6 từ config.
Chạy OCR từng frame.
Lưu line-level OCR raw.
```

Output `ppocr_raw.parquet`:

```text
video_id: string
frame_id: string
frame_number: int
frame_path: string
line_idx: int
bbox: list[int]       # [x1, y1, x2, y2]
poly: list[list[int]] # [[x,y], ...]
ocr_text: string
confidence: float
ocr_engine: string
runtime_ms: float|null
```

Acceptance:

```text
Có checkpoint mỗi N frames.
Có failed_jobs.jsonl.
Không mất kết quả nếu job bị dừng giữa chừng.
```

---

### 6.4. `risk_scoring.py`

Nhiệm vụ:

```text
Tính risk_score cho từng OCR line.
Không chỉ dựa vào confidence.
```

Feature đề xuất:

```text
low_confidence_score
suspicious_char_score
missing_diacritic_score
vietnamese_spell_risk_score
noise_text_score
temporal_disagreement_score
layout_prior_score
```

Pseudocode:

```python
def compute_ocr_risk(row):
    text = row["ocr_text"] or ""
    conf = float(row["confidence"])

    risk = 0.0

    if conf < 0.75:
        risk += 0.45
    elif conf < 0.90:
        risk += 0.20

    if has_suspicious_chars(text):
        risk += 0.35

    if has_vietnamese_spelling_risk(text):
        risk += 0.25

    if is_noise_text(text):
        risk += 0.15

    if has_temporal_disagreement(row):
        risk += 0.20

    return min(risk, 1.0)
```

Output `ocr_risk.parquet` thêm field:

```text
risk_score: float
need_vlm_line: bool
risk_reasons: list[string]
```

Acceptance:

```text
Các dòng lỗi dấu tiếng Việt nhưng confidence cao vẫn có thể bị flag.
Các dòng timestamp/logo/noise có thể skip VLM nếu rule cho phép.
```

---

### 6.5. `grouping.py`, `cropper.py`, `layout_classifier.py`

Nhiệm vụ:

```text
Group các OCR lines gần nhau thành group để crop và sửa theo ngữ cảnh.
Phân loại layout để quyết định có cần VLM không.
```

Region types:

```text
ticker
subtitle
signboard
logo
timestamp
table
document_block
scene_text
unknown
```

Group-level fields:

```text
video_id
frame_id
group_id
line_indices
merged_bbox
crop_path
crop_phash
raw_group_text
min_confidence
avg_confidence
max_risk_score
need_vlm_group
region_type
```

Quy tắc `need_vlm_group`:

```python
need_vlm_group = (
    max_risk_score >= cfg.risk.vlm_risk_threshold
    or any_line_has_suspicious_chars
    or region_type in ["signboard", "document_block", "table"]
) and region_type not in ["timestamp", "logo"]
```

Acceptance:

```text
Không crop full frame trừ debug mode.
Crop phải có padding nhưng không vượt biên ảnh.
Crop quá nhỏ phải upscale đến MIN_SIDE.
```

---

### 6.6. `vlm_prompt.py`

Nhiệm vụ:

```text
Tạo prompt correction ngắn, ép model trả JSON hợp lệ.
Không yêu cầu giải thích.
Không cho model thêm reasoning.
```

Prompt template:

```text
Bạn là hệ thống sửa lỗi OCR tiếng Việt.
Nhiệm vụ: dựa vào ảnh crop và danh sách raw OCR bên dưới, sửa lỗi dấu, chính tả và ký tự OCR.
Không thêm thông tin mới ngoài nội dung nhìn thấy trong ảnh.
Giữ số dòng output bằng số dòng input.
Chỉ trả về JSON hợp lệ theo schema:
{
  "lines": [
    {"line_idx": 0, "raw_text": "...", "corrected_text": "...", "confidence_note": "low|medium|high"}
  ]
}

Raw OCR lines:
[0] ...
[1] ...
```

Generation config:

```python
max_new_tokens = min(
    cfg.vlm.max_new_tokens_cap,
    cfg.vlm.max_new_tokens_base + n_lines * cfg.vlm.max_new_tokens_per_line
)

generation_config = {
    "max_new_tokens": max_new_tokens,
    "do_sample": False,
    "temperature": 0.0,
    "num_beams": 1,
}
```

Acceptance:

```text
parse_success_rate >= 95% trên sample benchmark.
Nếu JSON invalid, retry 1 lần với prompt ngắn hơn.
Nếu vẫn fail, fallback corrected_text = raw_text và ghi parse_error.
```

---

### 6.7. `vlm_loader.py`, `vlm_corrector.py`, `vlm_router.py`

Nhiệm vụ:

```text
Load model Vintern theo config.
Cho phép switch Vintern-3B-beta và Vintern-3B-R-beta.
Chỉ chạy VLM trên jobs cần sửa.
```

Interface đề xuất:

```python
class VLMCorrector:
    def __init__(self, model_id: str, quantization: str, device: str):
        ...

    def correct_group(self, image_crop, raw_lines, metadata) -> dict:
        ...
```

Router:

```python
def choose_vlm_model(group):
    if not cfg.vlm.allow_model_switch:
        return cfg.vlm.model_id

    if group.region_type in ["table", "document_block", "complex_signboard"]:
        return cfg.vlm.alternative_model_id

    if group.max_risk_score >= 0.85:
        return cfg.vlm.alternative_model_id

    return cfg.vlm.model_id
```

Production strategy:

```text
Giai đoạn 1: chạy một model duy nhất = Vintern-3B-beta.
Giai đoạn 2: benchmark R-beta trên nhóm khó.
Giai đoạn 3: bật hybrid routing nếu R-beta tăng accuracy đủ lớn.
```

Acceptance:

```text
Có thể đổi model bằng config mà không sửa code.
Có thể chạy benchmark 2 model trên cùng sample.
Có cache theo model_id để không lẫn kết quả giữa 2 model.
```

---

### 6.8. `merge_results.py`

Nhiệm vụ:

```text
Merge PP-OCR raw line-level với VLM corrected group-level.
Tạo frame-level OCR summary.
```

Line-level output:

```text
raw_text
corrected_text
normalized_text
confidence
risk_score
need_vlm
vlm_corrected
vlm_model
text_changed
parse_status
```

Frame-level OCR summary:

```text
ocr.raw_text
ocr.corrected_text
ocr.normalized_text
ocr.line_count
ocr.min_confidence
ocr.avg_confidence
ocr.vlm_corrected
ocr.vlm_model_ids
ocr.num_corrected_lines
```

Acceptance:

```text
Số line sau merge phải bằng số line raw OCR.
Nếu VLM trả thiếu dòng, fallback dòng thiếu = raw_text.
Không được làm mất bbox/poly.
```

---

## 7. Elasticsearch design

### 7.1. Index strategy

Index chính:

```text
video_frames_v1
```

Document granularity:

```text
1 Elasticsearch document = 1 frame
```

Lý do:

```text
Search thường cần trả về frame/video/timestamp.
OCR là feature text của frame.
OCR lines dùng nested để giữ bbox/poly và metadata line-level.
```

---

### 7.2. Frame document schema

```json
{
  "video_id": "L25_V001",
  "frame_id": "008",
  "frame_number": 8,
  "timestamp_ms": 320,
  "frame_path": "/data/frames/L25_V001/008.jpg",

  "image": {
    "width": 1280,
    "height": 720,
    "phash": "...",
    "scene_id": "scene_0003"
  },

  "ocr": {
    "raw_text": "...",
    "corrected_text": "...",
    "normalized_text": "...",
    "has_text": true,
    "line_count": 9,
    "min_confidence": 0.88,
    "avg_confidence": 0.95,
    "vlm_corrected": true,
    "vlm_model_ids": ["5CD-AI/Vintern-3B-beta"],
    "ocr_engine": "PP-OCRv6",
    "lines": [
      {
        "line_id": 0,
        "group_id": 1,
        "region_type": "signboard",
        "bbox": [571, 0, 1236, 270],
        "poly": [[571, 0], [1236, 0], [1236, 270], [571, 270]],
        "raw_text": "Nguyn Minh Châu",
        "corrected_text": "Nguyễn Minh Châu",
        "normalized_text": "nguyen minh chau",
        "confidence": 0.991,
        "risk_score": 0.65,
        "need_vlm": true,
        "vlm_corrected": true,
        "text_changed": true
      }
    ]
  },

  "quality": {
    "blur_score": 0.21,
    "brightness": 0.74,
    "is_duplicate": false
  },

  "created_at": "2026-06-25T20:00:00+07:00"
}
```

---

### 7.3. Elasticsearch mapping

File: `configs/es_mapping.json`

```json
{
  "settings": {
    "analysis": {
      "filter": {
        "vi_edge_ngram_filter": {
          "type": "edge_ngram",
          "min_gram": 2,
          "max_gram": 20
        }
      },
      "analyzer": {
        "vi_text": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": ["lowercase", "asciifolding"]
        },
        "vi_search_as_you_type": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": ["lowercase", "asciifolding", "vi_edge_ngram_filter"]
        }
      }
    }
  },
  "mappings": {
    "dynamic": false,
    "properties": {
      "video_id": { "type": "keyword" },
      "frame_id": { "type": "keyword" },
      "frame_number": { "type": "integer" },
      "timestamp_ms": { "type": "long" },
      "frame_path": { "type": "keyword", "index": false },

      "image": {
        "properties": {
          "width": { "type": "integer" },
          "height": { "type": "integer" },
          "phash": { "type": "keyword" },
          "scene_id": { "type": "keyword" }
        }
      },

      "ocr": {
        "properties": {
          "raw_text": {
            "type": "text",
            "analyzer": "vi_text"
          },
          "corrected_text": {
            "type": "text",
            "analyzer": "vi_text",
            "fields": {
              "keyword": { "type": "keyword", "ignore_above": 512 },
              "suggest": {
                "type": "text",
                "analyzer": "vi_search_as_you_type"
              }
            }
          },
          "normalized_text": {
            "type": "text",
            "analyzer": "vi_text"
          },
          "has_text": { "type": "boolean" },
          "line_count": { "type": "integer" },
          "min_confidence": { "type": "float" },
          "avg_confidence": { "type": "float" },
          "vlm_corrected": { "type": "boolean" },
          "vlm_model_ids": { "type": "keyword" },
          "ocr_engine": { "type": "keyword" },

          "lines": {
            "type": "nested",
            "properties": {
              "line_id": { "type": "integer" },
              "group_id": { "type": "integer" },
              "region_type": { "type": "keyword" },
              "bbox": { "type": "integer" },
              "poly": { "type": "integer" },
              "raw_text": { "type": "text", "analyzer": "vi_text" },
              "corrected_text": {
                "type": "text",
                "analyzer": "vi_text",
                "fields": {
                  "keyword": { "type": "keyword", "ignore_above": 256 }
                }
              },
              "normalized_text": { "type": "text", "analyzer": "vi_text" },
              "confidence": { "type": "float" },
              "risk_score": { "type": "float" },
              "need_vlm": { "type": "boolean" },
              "vlm_corrected": { "type": "boolean" },
              "text_changed": { "type": "boolean" }
            }
          }
        }
      },

      "quality": {
        "properties": {
          "blur_score": { "type": "float" },
          "brightness": { "type": "float" },
          "is_duplicate": { "type": "boolean" }
        }
      },

      "created_at": { "type": "date" }
    }
  }
}
```

---

## 8. Search query examples

### 8.1. Fuzzy OCR search

```json
GET video_frames_v1/_search
{
  "size": 20,
  "_source": [
    "video_id",
    "frame_id",
    "timestamp_ms",
    "frame_path",
    "ocr.corrected_text"
  ],
  "query": {
    "multi_match": {
      "query": "nguen minh chau",
      "fields": [
        "ocr.corrected_text^3",
        "ocr.normalized_text^2",
        "ocr.raw_text"
      ],
      "fuzziness": "AUTO"
    }
  }
}
```

### 8.2. Nested line-level search với inner_hits

```json
GET video_frames_v1/_search
{
  "size": 10,
  "query": {
    "nested": {
      "path": "ocr.lines",
      "query": {
        "match": {
          "ocr.lines.corrected_text": {
            "query": "nguyen minh chau",
            "fuzziness": "AUTO"
          }
        }
      },
      "inner_hits": {
        "size": 3,
        "_source": [
          "ocr.lines.line_id",
          "ocr.lines.corrected_text",
          "ocr.lines.bbox",
          "ocr.lines.region_type"
        ]
      }
    }
  }
}
```

---

## 9. Runtime strategy cho 1 GPU A5000

### 9.1. Không nên load PP-OCR và Vintern đồng thời nếu VRAM căng

Khuyến nghị chạy theo stage:

```text
Stage 1: Load PP-OCRv6 → chạy toàn bộ OCR → unload/clear GPU.
Stage 2: Load Vintern → chạy correction jobs → unload/clear GPU.
Stage 3: Build ES docs CPU/I/O.
```

### 9.2. Restartable pipeline

Mọi stage phải có resume:

```text
Nếu ppocr_raw.parquet đã tồn tại thì không chạy lại PP-OCR trừ khi --force.
Nếu vlm_corrected.parquet đã có group_id thì skip group đó.
Nếu es_documents.jsonl đã có thì bulk index tiếp theo batch.
```

### 9.3. Ưu tiên tối ưu chi phí

Thứ tự tối ưu:

```text
1. Frame-level dedup.
2. OCR raw lưu parquet, không OCR lại.
3. Risk-based gating thay vì gọi VLM toàn bộ.
4. Crop-level cache.
5. Dynamic max_new_tokens.
6. Strict JSON output để giảm retry/parse fail.
7. Chạy benchmark trước khi chọn R-beta cho production.
```

---

## 10. Evaluation plan

### 10.1. Sample benchmark

Tạo sample gồm:

```text
100 frames random.
100 frames có nhiều text.
100 frames low-confidence.
100 frames high-confidence nhưng có suspicious chars.
100 frames từ các scene khác nhau.
```

### 10.2. Metrics

```text
PP-OCR runtime sec/frame
VLM runtime sec/group
VLM cache hit rate
VLM parse success rate
Text changed rate
Manual correction accuracy
Search Recall@K trên query OCR
Storage size/index size
Bulk indexing docs/sec
```

### 10.3. Model comparison

Bảng output `model_benchmark.csv`:

```text
model_id,num_groups,avg_latency_sec,p95_latency,gpu_mem_gb,json_valid_rate,manual_accuracy,notes
5CD-AI/Vintern-3B-beta,...
5CD-AI/Vintern-3B-R-beta,...
erax-ai/EraX-VL-2B-V1.5,...
```

Decision rule:

```text
Nếu accuracy của R-beta cao hơn < 2% nhưng latency tăng đáng kể: chọn 3B-beta.
Nếu R-beta cao hơn >= 5% trên nhóm khó: dùng hybrid routing.
Nếu R-beta parse fail nhiều do reasoning output: chỉ dùng 3B-beta cho production.
Nếu EraX-VL-2B-V1.5 đạt accuracy tốt hơn Vintern với latency/VRAM chấp nhận được: thêm vào router như model alternative cho nhóm OCR khó.
```

---

## 11. CLI flow đề xuất

### 11.1. Chạy từng stage

```bash
python scripts/01_build_frame_registry.py --config configs/server_a5000.yaml
python scripts/02_run_ppocr.py --config configs/server_a5000.yaml
python scripts/03_build_vlm_jobs.py --config configs/server_a5000.yaml
python scripts/04_run_vlm_correction.py --config configs/server_a5000.yaml
python scripts/05_merge_features.py --config configs/server_a5000.yaml
python scripts/06_build_es_documents.py --config configs/server_a5000.yaml
python scripts/07_bulk_index_es.py --config configs/server_a5000.yaml
python scripts/08_search_demo.py --config configs/server_a5000.yaml --query "nguyen minh chau"
```

### 11.2. Chạy full pipeline

```bash
python scripts/run_full_pipeline.py --config configs/server_a5000.yaml
```

### 11.3. Benchmark model Vintern

```bash
python scripts/run_vlm_benchmark.py \
  --config configs/server_a5000.yaml \
  --models 5CD-AI/Vintern-3B-beta 5CD-AI/Vintern-3B-R-beta erax-ai/EraX-VL-2B-V1.5 \
  --sample_groups 500
```

---

## 12. Acceptance criteria tổng thể

Pipeline được xem là đạt khi:

```text
[ ] Có thể chạy lại từng stage mà không mất kết quả cũ.
[ ] Có thể switch Vintern-3B-beta / Vintern-3B-R-beta bằng config.
[ ] VLM chỉ chạy trên selected groups, không chạy toàn bộ frame.
[ ] Cache VLM hoạt động và giảm số lần inference lặp lại.
[ ] Raw OCR, corrected OCR, normalized OCR đều được lưu.
[ ] Elasticsearch index có OCR nested lines.
[ ] Fuzzy query tìm được text sai dấu/không dấu/phát âm gần đúng.
[ ] Có benchmark latency/accuracy cho 2 model Vintern.
[ ] Có benchmark latency/accuracy cho EraX-VL-2B-V1.5.
[ ] Có failed_jobs.jsonl để debug.
[ ] Có sample_review.csv để review thủ công chất lượng correction.
```

---

## 13. Ưu tiên coding cho agent

Thứ tự code đề xuất:

```text
1. config.py + default.yaml/server_a5000.yaml
2. frame_registry.py
3. ppocr_engine.py + ppocr_runner.py
4. normalize_text.py
5. risk_scoring.py
6. grouping.py + cropper.py
7. vlm_prompt.py + vlm_parser.py
8. vlm_loader.py + vlm_corrector.py
9. vlm_cache.py
10. vlm_router.py
11. merge_results.py
12. es_document_builder.py
13. es_schema.py + es_bulk_indexer.py
14. search_demo.py
15. benchmark.py + validation.py
```

Minimum viable version:

```text
MVP 1:
  PP-OCRv6 raw → risk scoring → grouping → Vintern-3B-beta correction → merge parquet.

MVP 2:
  Add VLM cache + switch model + benchmark 2 models.

MVP 3:
  Build ES documents + mapping + bulk index + fuzzy search.

MVP 4:
  Optimize scale: frame dedup, ROI dedup, resume, runtime stats.
```

---

## 14. Notes for agent

Không implement theo hướng:

```text
Không đưa full frame vào VLM cho mọi frame.
Không sửa trực tiếp notebook demo thành spaghetti code.
Không overwrite raw OCR.
Không hard-code model_id trong nhiều file.
Không dùng confidence threshold đơn lẻ làm điều kiện duy nhất gọi VLM.
Không gọi R-beta mặc định cho toàn bộ production nếu chưa benchmark.
```

Nên implement theo hướng:

```text
Config-driven.
Stage-based.
Resume-friendly.
Cache-first.
Strict JSON output.
Frame-level ES document.
Nested OCR lines.
Benchmark trước khi chọn model production.
```
