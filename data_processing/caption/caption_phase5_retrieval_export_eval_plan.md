# Caption Phase 5 Retrieval Export + Evaluation Plan

## 0. Trang thai dau vao

Phase 0/1/2/3/4 da pass tren `L22_V012`:

```text
frame_evidence.jsonl        = 283
shot_evidence.jsonl         = 102
frame_index.jsonl           = 283
shot_index.jsonl            = 102
event_step_index.jsonl      = 102
compact_search_index.jsonl  = 385
trake_text coverage         = 102/102 shot docs
```

Phase 5 khong nen goi VLM/LLM moi ngay. Buoc hop ly tiep theo la bien output caption hien tai thanh bo du lieu retrieval-ready va co local evaluator de do duoc chat luong search truoc khi benchmark VLM fuser.

---

## 1. Muc tieu Phase 5

Phase 5 tao lop search/evaluation dung chung cho:

- Frame retrieval: query visual/OCR/object/audio tra ve frame.
- Shot retrieval: query clip/temporal context tra ve shot.
- TRAKE retrieval: query nhieu buoc co thu tu tra ve event-step/shot sequence.
- Elasticsearch export: tao mapping va bulk JSONL co document id on dinh.
- Offline benchmark: chay BM25-like lexical baseline khong can ES de test nhanh tren server.

Sau Phase 5, ta co baseline search de so sanh:

```text
A: evidence + template caption + shot caption + trake_text
B: A + FuseCap/BLIP initial caption
C: A/B + VLM fuser
```

---

## 2. Output sau Phase 5

Ghi trong:

```text
caption/<video_id>/
├── exports/
│   ├── es_mapping_compact_search.json
│   ├── es_mapping_event_steps.json
│   ├── es_bulk_compact_search.jsonl
│   ├── es_bulk_event_steps.jsonl
│   └── retrieval_corpus.jsonl
├── eval/
│   ├── retrieval_queries.jsonl
│   ├── retrieval_results.jsonl
│   ├── retrieval_eval_report.json
│   └── retrieval_eval_report.md
└── indexes/
    └── compact_search_index.jsonl
```

Expected for `L22_V012`:

```text
retrieval_corpus rows          = 487
  compact docs                 = 385
  event_step docs              = 102
es_bulk_compact_search lines   = 770  # action/doc pairs
es_bulk_event_steps lines      = 204  # action/doc pairs
retrieval queries              >= 12
retrieval results              = query_count * top_k
blocking warnings              = 0
```

Note: `retrieval_corpus.jsonl` gom compact docs + event_step docs de local search offline. Khi import ES that, co the dung 2 index rieng:

```text
aic_caption_compact_v1
aic_caption_event_step_v1
```

---

## 3. File/code se them

```text
data_processing/caption/
├── run_retrieval_export_eval.py
└── caption_pipeline/
    ├── retrieval_export.py
    ├── retrieval_eval.py
    ├── retrieval_report.py
    └── retrieval_queries.py
```

Khong nen gop vao `run_caption_pipeline.py` o Phase 5A, vi:

- Caption pipeline da pass va nen giu on dinh.
- Retrieval export/eval la buoc sau, co the chay lap lai nhieu lan.
- Sau nay co the scale batch va import ES ma khong can regenerate caption.

---

## 4. CLI de code

Script moi:

```bash
python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --top-k 20 \
  --write-es-bulk \
  --write-default-queries
```

Tham so:

```text
--video-id
--caption-dir
--compact-index optional, default caption/<video_id>/indexes/compact_search_index.jsonl
--event-step-index optional, default caption/<video_id>/captions/event_step_index.jsonl
--queries-jsonl optional, default caption/<video_id>/eval/retrieval_queries.jsonl
--top-k default 20
--write-es-bulk
--write-default-queries
--strict
```

`--strict` se return non-zero neu:

- Thieu input required.
- Corpus rong.
- Duplicate `document_id`.
- Event-step count khong bang shot count neu co du lieu shot.
- Query test khong co hit nao.

---

## 5. Retrieval corpus schema

`retrieval_corpus.jsonl` la schema noi bo de local eval.

```json
{
  "schema_version": "caption_retrieval_corpus_v1",
  "document_id": "compact:shot:L22_V012_shot_0032",
  "source": "compact_search_index",
  "unit_type": "shot",
  "video_id": "L22_V012",
  "unit_id": "L22_V012_shot_0032",
  "frame_id": "",
  "shot_id": "L22_V012_shot_0032",
  "event_id": "",
  "timestamp_sec": 120.0,
  "start_sec": 120.0,
  "end_sec": 136.5,
  "search_text": "...",
  "fields": {
    "caption_text": "...",
    "temporal_caption": "...",
    "ocr_text": "...",
    "audio_text": "...",
    "object_text": "...",
    "scene_text": "...",
    "trake_text": "..."
  },
  "terms": ["..."],
  "boosts": {
    "caption": 2.0,
    "ocr": 2.5,
    "audio": 1.4,
    "object": 1.6,
    "trake": 2.2
  }
}
```

Document id convention:

```text
compact:frame:<frame_id>
compact:shot:<shot_id>
event_step:<event_id>
```

Neu compact doc cu chua co `document_id`, Phase 5 export tu sinh id on dinh, khong sua file goc.

---

## 6. Elasticsearch export

### 6.1. Bulk JSONL format

`es_bulk_compact_search.jsonl`:

```json
{"index":{"_index":"aic_caption_compact_v1","_id":"compact:shot:L22_V012_shot_0032"}}
{"schema_version":"caption_compact_search_v1","unit_type":"shot","unit_id":"L22_V012_shot_0032","video_id":"L22_V012","all_text":"..."}
```

`es_bulk_event_steps.jsonl`:

```json
{"index":{"_index":"aic_caption_event_step_v1","_id":"event_step:L22_V012_event_0032"}}
{"schema_version":"caption_event_step_index_v1","event_id":"L22_V012_event_0032","shot_id":"L22_V012_shot_0032","trake_text":"..."}
```

### 6.2. Mapping compact search

Fields chinh:

```text
keyword:
  schema_version, unit_type, unit_id, video_id, frame_id

float:
  timestamp_sec, start_sec, end_sec

text:
  caption_text
  ocr_text
  audio_text
  object_text
  scene_text
  trake_text
  all_text
```

Analyzer Phase 5A:

- Neu ES co Vietnamese analyzer thi dung custom analyzer sau.
- Neu chua co plugin, dung `standard` + lowercase de ingest truoc.
- Khong hard-code analyzer phu thuoc plugin vao bulk writer.

### 6.3. Mapping event steps

Fields chinh:

```text
keyword:
  schema_version, document_id, video_id, event_id, shot_id
  action_state, temporal_role
  actors, actions, objects_involved

integer:
  step_order

float:
  start_sec, end_sec

text:
  event_caption
  current_observation
  before_context
  after_context
  scene
  trake_text
  trake_text_search
```

---

## 7. Local lexical evaluator

Phase 5A khong can ES. Ta implement scorer de test nhanh:

```text
score =
  field_match_score
+ bm25_like_term_score
+ unit_type_boost
+ temporal_bonus
```

### 7.1. Normalize query

Dung chung logic:

- lowercase
- remove accents
- `đ -> d`
- strip punctuation
- split tokens
- bo token do dai < 2

### 7.2. Field weights

Default:

```yaml
frame:
  caption_text: 2.0
  ocr_text: 2.5
  audio_text: 1.2
  object_text: 1.8
  scene_text: 1.0
  all_text: 0.8

shot:
  caption_text: 2.2
  trake_text: 2.0
  ocr_text: 2.0
  audio_text: 1.6
  object_text: 1.5
  scene_text: 1.0
  all_text: 0.8

event_step:
  trake_text: 2.8
  current_observation: 2.2
  event_caption: 2.0
  before_context: 0.8
  after_context: 0.8
  scene: 0.8
```

### 7.3. Query routing

Heuristic route:

```text
contains "chữ", "text", "logo", "bảng", "tiêu đề"
-> route = ocr

contains "nói", "phát biểu", "âm thanh", "giọng", "thảo luận"
-> route = audio

contains "người", "xe", "màn hình", object nouns
-> route = object_visual

contains "đầu tiên", "sau đó", "cuối cùng", "trước khi", "tiếp theo"
-> route = trake

otherwise
-> route = general
```

Route chi anh huong field weights, khong filter cung.

---

## 8. Default query set

Neu `--write-default-queries`, tao `retrieval_queries.jsonl` gom it nhat 12 query de sanity check.

Schema:

```json
{
  "query_id": "q_ocr_001",
  "query": "cảnh có chữ hội nghị chuyển đổi số",
  "route": "ocr",
  "target_unit_types": ["frame", "shot"],
  "expected_terms_any": ["hoi nghi", "chuyen doi so"],
  "notes": "OCR-oriented query"
}
```

Loai query:

```text
OCR:
  - cảnh có chữ hội nghị
  - cảnh có chữ thành phố

Object/visual:
  - người đứng trước màn hình
  - nhiều người trong hội trường

Audio/topic:
  - đoạn nói về chuyển đổi số
  - phát biểu về quản lý hoặc vận hành

Shot/temporal:
  - đoạn người trình bày trước màn hình
  - cảnh chuyển sang một bối cảnh khác

TRAKE:
  - đầu tiên bắt đầu cảnh, sau đó tiếp tục trình bày
  - cuối cùng kết thúc hoặc chuyển sang phần khác
```

Do L22_V012 chua co ground truth query chinh thuc, Phase 5A chi dung `expected_terms_any` va manual inspection fields, khong coi day la benchmark khoa hoc cuoi cung.

---

## 9. Retrieval results schema

`retrieval_results.jsonl`:

```json
{
  "query_id": "q_ocr_001",
  "rank": 1,
  "score": 12.34,
  "document_id": "compact:frame:L22_V012_000123",
  "unit_type": "frame",
  "video_id": "L22_V012",
  "unit_id": "L22_V012_000123",
  "shot_id": "",
  "event_id": "",
  "start_sec": 123.4,
  "end_sec": 123.4,
  "matched_fields": ["ocr_text", "all_text"],
  "matched_terms": ["hoi", "nghi"],
  "snippet": "..."
}
```

---

## 10. Report

`retrieval_eval_report.json`:

```json
{
  "video_id": "L22_V012",
  "num_corpus_docs": 487,
  "num_queries": 12,
  "top_k": 20,
  "num_queries_with_hits": 12,
  "route_counts": {
    "ocr": 2,
    "object_visual": 2,
    "audio": 2,
    "general": 4,
    "trake": 2
  },
  "unit_type_hits_at_1": {
    "frame": 5,
    "shot": 4,
    "event_step": 3
  },
  "warnings": []
}
```

Markdown report can include:

- Summary counts.
- Top 5 results per query.
- Matched fields.
- Snippets.
- Warnings.

---

## 11. TRAKE local search behavior

For query co nhieu buoc:

```text
đầu tiên ... sau đó ... cuối cùng ...
```

Phase 5A tach bang keyword:

```text
["đầu tiên", "sau đó", "cuối cùng", "tiếp theo", "trước khi"]
```

Then:

1. Retrieve top-K event_step cho tung subquery.
2. Tao candidate chain cung `video_id`.
3. Chi giu chain co `step_order` tang dan.
4. Score chain:

```text
chain_score =
  mean(step_scores)
+ temporal_order_bonus
- duration_penalty
```

Output Phase 5A co the ghi chung vao `retrieval_results.jsonl` voi:

```json
{
  "unit_type": "event_chain",
  "document_id": "event_chain:L22_V012:q_trake_001:0001",
  "event_ids": ["...", "..."],
  "shot_ids": ["...", "..."],
  "start_sec": 10.0,
  "end_sec": 40.0
}
```

Neu code Phase 5A qua dai, event_chain co the de Phase 5B; nhung `event_step` search rieng phai co trong Phase 5A.

---

## 12. Acceptance criteria

Phase 5 pass khi:

```text
retrieval_corpus rows = compact rows + event_step rows
compact docs exported to ES bulk = compact rows
event steps exported to ES bulk = event_step rows
no duplicate retrieval document_id
no empty search_text for docs with useful fields
all default queries have at least 1 hit
retrieval_eval_report.json exists
retrieval_eval_report.md exists
strict mode returns 0
```

For `L22_V012`:

```text
retrieval_corpus rows = 487
es_bulk_compact_search docs = 385
es_bulk_event_steps docs = 102
queries_with_hits = num_queries
warnings = 0 or only non-blocking warnings
```

---

## 13. Server run command after coding

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --top-k 20 \
  --write-es-bulk \
  --write-default-queries \
  --strict
```

Check:

```bash
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/exports/retrieval_corpus.jsonl
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/exports/es_bulk_compact_search.jsonl
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/exports/es_bulk_event_steps.jsonl
cat /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/eval/retrieval_eval_report.json
```

Expected:

```text
retrieval_corpus.jsonl        = 487
es_bulk_compact_search.jsonl  = 770
es_bulk_event_steps.jsonl     = 204
num_queries_with_hits         = num_queries
```

---

## 14. Sau Phase 5

Neu Phase 5 pass:

1. Import ES optional va test query that.
2. Them query set co ground truth tu AIC/TRAKE neu co.
3. Moi benchmark FuseCap/BLIP initial caption hoac VLM fuser.
4. So sanh voi report Phase 5:

```text
template text baseline vs FuseCap initial caption vs VLM fuser
```

Metric so sanh:

```text
Hit@K
MRR
manual relevance
latency/query
GPU/API cost
hallucination rate
```
