# Caption Phase 3 Shot Captioner Plan

## 0. Phase 2 output gate

Phase 3 chi bat dau sau khi Phase 0/1/2 da pass cho `L22_V012`.

Ket qua server gan nhat:

```text
frame_evidence rows       = 283
shot_evidence rows        = 102
frame_index rows          = 283
compact_search_index rows = 385
num_caption_generated     = 283
num_caption_empty         = 0
num_fallback_used         = 0
caption_mode_counts       = template: 283
duplicate_document_ids    = 0
duplicate_frame_ids       = 0
```

Compact index hien tai:

```text
frame docs = 283, da co caption_text
shot docs  = 102, chua co caption_text
```

Ket luan: co the tiep tuc Phase 3 de sinh caption cho shot-level.

---

## 1. Muc tieu Phase 3

Phase 3 tao shot-level caption theo huong ReCap-style temporal captioning, nhung implementation dau tien van la deterministic/template baseline.

Input trong pipeline:

```text
shot_evidence                # tu Phase 1, in-memory
frame_index                  # tu Phase 2, in-memory
compact_search_index docs    # frame docs da co caption_text neu bat rebuild
```

Input file tu output neu can debug:

```text
caption/<video_id>/evidence/shot_evidence.jsonl
caption/<video_id>/captions/frame_index.jsonl
caption/<video_id>/indexes/compact_search_index.jsonl
```

Output moi:

```text
caption/<video_id>/captions/shot_index.jsonl
caption/<video_id>/reports/shot_caption_report.json
caption/<video_id>/reports/shot_caption_report.md
```

Output co the cap nhat:

```text
caption/<video_id>/indexes/compact_search_index.jsonl
```

Sau Phase 3, compact index van giu 385 docs, nhung shot docs cung co `caption_text`.

---

## 2. Pham vi Phase 3

### 2.1. Nen lam trong Phase 3A

- Template shot captioner khong dung model.
- Dung `shot_evidence` + frame captions cua representative frames.
- Tao `shot_index.jsonl`.
- Tao report chat luong shot captions.
- Rebuild compact index de shot docs co `caption_text`.
- Them memory text deterministic ngan de lam nen cho Phase 3B/LLM.

### 2.2. Chua lam trong Phase 3A

- Chua goi LLM that.
- Chua goi VLM/image captioner.
- Chua lam event-step/TRAKE.
- Chua thay doi schema Phase 0/1/2.
- Chua viet Elasticsearch mapping final.

### 2.3. Phase 3B sau nay

Sau khi template pass, moi them:

```text
--shot-caption-mode llm
```

LLM du kien:

```text
Qwen/Qwen2.5-7B-Instruct
```

Nhung mode `llm` phai co warning/fallback ro rang neu chua implement, giong fix o Phase 2.

---

## 3. Schema `shot_index.jsonl`

Moi record:

```json
{
  "schema_version": "caption_shot_index_v1",
  "document_id": "caption_shot:L22_V012_shot_0001",
  "unit_type": "shot",
  "video_id": "L22_V012",
  "shot_id": "L22_V012_shot_0001",
  "start_sec": 0.0,
  "end_sec": 3.0,
  "representative_frame_ids": ["L22_V012_001", "L22_V012_002"],
  "frame_count": 2,

  "caption_text": "Shot shows ...",
  "temporal_caption": "At the beginning of the video, ...",
  "caption_text_search": "shot shows ...",
  "temporal_caption_search": "at the beginning ...",
  "caption_terms": ["..."],

  "evidence_text": {
    "merged_ocr_text": "giay HTV 7 HD ...",
    "merged_audio_text": "Chao mung quy vi ...",
    "object_text": "2 person 1 chair",
    "scene_text": "city sky sun water",
    "representative_frame_caption_text": "Scene shows ... || Scene shows ..."
  },

  "memory": {
    "memory_before": "",
    "memory_after": "opening segment introducing the 60 giay program",
    "previous_shot_id": "",
    "next_shot_id": "L22_V012_shot_0002"
  },

  "search_fields": {
    "all_text": "ocr audio objects scene frame captions shot caption temporal caption",
    "caption_boost_text": "shot caption temporal caption",
    "frame_caption_text": "representative frame captions",
    "visual_text": "objects scene",
    "spoken_text": "audio",
    "onscreen_text": "ocr",
    "temporal_text": "memory_before temporal_caption memory_after"
  },

  "quality": {
    "has_caption": true,
    "caption_mode": "template",
    "caption_model": "",
    "prompt_version": "",
    "used_ocr": true,
    "used_audio": true,
    "used_objects": true,
    "used_scene": true,
    "used_frame_captions": true,
    "fallback_used": false,
    "num_representative_frames": 2,
    "num_representative_frame_captions": 2,
    "warnings": []
  }
}
```

Ghi chu:

- `caption_text`: caption shot doc chinh cho search.
- `temporal_caption`: caption co boi canh thoi gian/memory.
- `caption_text_search` va `temporal_caption_search`: lowercase + remove accents + `đ -> d`.
- `memory_before/memory_after`: deterministic rolling memory, khong duoc them thong tin ngoai evidence.
- `representative_frame_caption_text`: chi ghep captions cua representative frames, khong ghep 100% frame captions de tranh qua dai.

---

## 4. Code layout can them

```text
data_processing/caption/
├── caption_phase3_shot_captioner_plan.md
├── run_caption_pipeline.py
└── caption_pipeline/
    ├── shot_captioner.py
    ├── shot_index.py
    ├── shot_caption_report.py
    └── prompts/
        └── shot_caption_prompt.txt
```

Can sua them:

```text
caption_pipeline/config.py
caption_pipeline/schemas.py
caption_pipeline/caption_index.py
run_caption_pipeline.py
```

---

## 5. Config va CLI

### 5.1. `config.py`

Them config:

```python
@dataclass
class ShotCaptionConfig:
    caption_mode: str = "template"  # "template" or "llm"
    max_ocr_chars: int = 220
    max_audio_chars: int = 360
    max_frame_caption_chars: int = 360
    max_caption_chars: int = 520
    max_temporal_caption_chars: int = 620
    max_memory_chars: int = 500
```

Them vao `PipelineConfig`:

```python
enable_shot_captions: bool = False
shot_caption: ShotCaptionConfig = field(default_factory=ShotCaptionConfig)
```

### 5.2. CLI

Them flags:

```bash
--enable-shot-captions
--shot-caption-mode template
```

Giu lai flag hien co:

```bash
--rebuild-compact-with-captions
```

Trong Phase 3, neu flag nay bat:

- frame docs: cap nhat `caption_text` tu `frame_index`
- shot docs: cap nhat `caption_text` tu `shot_index`

Khong nen them `--rebuild-compact-with-shot-captions` truoc, de CLI khong bi roi.

### 5.3. Dependency giua Phase 2 va Phase 3

Phase 3 can `frame_index`.

Neu user chay:

```bash
--enable-shot-captions
```

ma khong bat:

```bash
--enable-frame-captions
```

thi code nen:

- Cach don gian: return loi ro rang.
- Cach tot hon sau nay: doc `--frame-index-jsonl` tu output co san.

Phase 3A khuyen nghi cach don gian:

```text
if enable_shot_captions and not enable_frame_captions:
    error: Phase 3 requires --enable-frame-captions in this integrated pipeline.
```

---

## 6. `shot_captioner.py`

Trach nhiem:

```text
shot_evidence + frame_index -> shot caption records
```

API de xuat:

```python
def fuse_shot_captions(
    shot_evidence: list[dict],
    frame_index: list[dict],
    cfg: PipelineConfig,
) -> list[dict]:
    ...
```

Moi caption record trung gian:

```json
{
  "shot_id": "L22_V012_shot_0001",
  "caption_text": "...",
  "temporal_caption": "...",
  "memory_before": "...",
  "memory_after": "...",
  "caption_mode": "template",
  "caption_model": "",
  "prompt_version": "",
  "used_ocr": true,
  "used_audio": true,
  "used_objects": true,
  "used_scene": true,
  "used_frame_captions": true,
  "fallback_used": false,
  "warnings": []
}
```

### 6.1. Template caption rule

Thu tu evidence:

1. representative frame captions
2. merged object counts
3. merged scene tags
4. merged OCR text
5. merged audio text

Template:

```text
Shot covers [start_sec]-[end_sec] seconds. Scene shows [scene phrase]. Visible objects include [object phrase]. Representative frames show [frame caption phrase]. On-screen text mentions "[ocr phrase]". Spoken content mentions [audio phrase].
```

Sau do clean:

- remove double punctuation
- truncate `caption_text` max 520 chars
- ensure ends with `.`, `!`, or `?`

### 6.2. Temporal caption rule

Template:

```text
[temporal position phrase] [caption_text] [memory reference if useful]
```

Temporal position:

- Shot 1: `At the beginning of the video,`
- Middle shots: `Then,`
- Last shot: `Near the end,`

Khong nen ghi memory reference neu `memory_before` rong.

### 6.3. Memory rule

Memory deterministic:

```text
memory_before = previous memory_after
memory_after  = compact summary of current shot + previous memory
```

Memory after nen chi gom:

- top scene tags
- top object labels
- short audio topic
- short OCR topic

Max 500 chars.

Khong copy nguyen audio dai vao memory.

### 6.4. LLM placeholder

Neu `--shot-caption-mode llm` ma chua implement:

- Van tao template caption.
- Set `caption_mode = "fallback"`.
- Set `fallback_used = true`.
- Them warning:

```text
llm_mode_not_implemented_using_template
```

Dieu nay tranh hieu nham la da chay LLM that.

---

## 7. `shot_index.py`

Trach nhiem:

```text
shot_evidence + frame_index + shot caption records -> shot_index.jsonl
```

API:

```python
def build_shot_index(
    shot_evidence: list[dict],
    frame_index: list[dict],
    shot_caption_records: list[dict],
) -> list[dict]:
    ...
```

Can build:

- `caption_text_search`
- `temporal_caption_search`
- `caption_terms`
- `evidence_text`
- `search_fields`
- `quality`

Frame map:

```python
frame_map = {row["canonical_frame_id"]: row for row in frame_index}
```

Representative captions:

```python
rep_caption_texts = [
    frame_map[frame_id]["caption_text"]
    for frame_id in shot["representative_frame_ids"]
    if frame_id in frame_map
]
```

Warning neu missing representative captions:

```text
missing_representative_frame_caption:<frame_id>
```

---

## 8. Compact index rebuild sau Phase 3

Hien co:

```python
rebuild_compact_with_captions(compact_docs, caption_records)
```

Phase 3 nen refactor thanh mot trong hai cach:

### Cach A - Them ham moi

```python
def rebuild_compact_with_frame_and_shot_captions(
    compact_docs: list[dict],
    frame_caption_records: list[dict],
    shot_caption_records: list[dict],
) -> list[dict]:
    ...
```

### Cach B - Giu ham cu va them optional param

```python
def rebuild_compact_with_captions(
    compact_docs: list[dict],
    frame_caption_records: list[dict] | None = None,
    shot_caption_records: list[dict] | None = None,
) -> list[dict]:
    ...
```

Khuyen nghi: Cach A ro rang hon, it pha logic Phase 2.

Update rules:

- For `unit_type == "frame"`:
  - key = `unit_id`
  - lookup frame caption by `canonical_frame_id`
- For `unit_type == "shot"`:
  - key = `unit_id`
  - lookup shot caption by `shot_id`

Rebuild `all_text`:

```text
ocr_text + audio_text + object_text + scene_text + caption_text + trake_text
```

Sau Phase 3:

```text
compact_search_index rows       = 385
frame docs with caption_text    = 283
shot docs with caption_text     = 102
shot docs with empty trake_text  = 102
```

---

## 9. `shot_caption_report.py`

Report JSON:

```json
{
  "video_id": "L22_V012",
  "created_at": "...",
  "num_shots": 102,
  "num_caption_generated": 102,
  "num_caption_empty": 0,
  "num_temporal_caption_generated": 102,
  "num_temporal_caption_empty": 0,
  "num_fallback_used": 0,
  "coverage": {
    "ocr": 0,
    "audio": 0,
    "objects": 0,
    "scene": 0,
    "frame_captions": 0
  },
  "caption_mode_counts": {
    "template": 102
  },
  "duplicate_document_ids": 0,
  "duplicate_shot_ids": 0,
  "num_shot_warnings": 0,
  "shot_warnings": [],
  "summary_warnings": []
}
```

Report markdown:

- ASCII-safe.
- Dung `[PASS]` / `[FAIL]`, khong dung emoji.
- Cap warnings first 50 giong Phase 2.

---

## 10. `run_caption_pipeline.py` flow

Thu tu sau khi sua:

```text
Phase 0: load + align
Phase 0: build frame_evidence
Phase 1: build shot_evidence
Phase 1: build compact index baseline
write Phase 0/1 outputs

if enable_frame_captions:
    Phase 2: fuse frame captions
    Phase 2: build frame_index
    write frame_index + frame_caption_report

if enable_shot_captions:
    require frame_index
    Phase 3: fuse shot captions
    Phase 3: build shot_index
    write shot_index + shot_caption_report

if rebuild_compact_with_captions:
    update compact docs with available frame captions and shot captions
    write compact_search_index.jsonl once after caption phases
```

Ghi chu quan trong:

- Hien code Phase 2 dang write compact index truoc, roi rebuild ngay sau Phase 2.
- Phase 3 nen tranh write compact 2 lan neu co the.
- Cach de code it rui ro:
  - Giu behavior Phase 2 hien tai.
  - Sau Phase 3, rebuild compact lan nua voi shot captions.
  - Sau nay refactor de write 1 lan.

---

## 11. Prompt file cho LLM sau nay

File:

```text
caption_pipeline/prompts/shot_caption_prompt.txt
```

Noi dung placeholder:

```text
You are a shot-level caption generator for video retrieval.

Given shot evidence and representative frame captions, generate:
1. caption_text: concise shot caption
2. temporal_caption: caption with temporal context
3. memory_after: compact memory for the next shot

Rules:
- Use only provided evidence.
- Do not mention metadata, JSON field names, confidence, bbox, or debug info.
- Summarize audio/OCR; do not copy long transcript verbatim.
- Keep caption_text under 520 characters.
- Keep temporal_caption under 620 characters.
- Output a single JSON object.

Expected JSON:
{
  "caption_text": "...",
  "temporal_caption": "...",
  "memory_after": "..."
}
```

Khong can goi prompt nay trong Phase 3A.

---

## 12. Acceptance checklist

Phase 3 dat chuan khi:

```text
shot_index rows = 102
shot captions generated = 102
shot caption empty = 0
temporal caption generated = 102
temporal caption empty = 0
duplicate document ids = 0
duplicate shot ids = 0
representative frame captions missing = 0
compact_search_index rows = 385
frame docs with caption_text = 283
shot docs with caption_text = 102
frame docs with trake_text empty = 283
shot docs with trake_text empty = 102
```

Quality checks nen them:

```text
max caption_text length <= 520
max temporal_caption length <= 620
caption_text_search has no Vietnamese "đ"
no double period ".." in caption_text
num_shot_warnings = 0 for template run
```

---

## 13. Lenh chay sau khi code Phase 3

Server command du kien:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_caption_pipeline.py \
  --video-id L22_V012 \
  --ocr-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs.jsonl \
  --object-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/object_detection/L22_V012_objects.jsonl \
  --audio-features /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/audio/L22_V012/features/audio_features.jsonl \
  --output-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --frame-window-before 5 \
  --frame-window-after 5 \
  --shot-max-gap-sec 5 \
  --shot-max-duration-sec 20 \
  --enable-frame-captions \
  --caption-mode template \
  --enable-shot-captions \
  --shot-caption-mode template \
  --rebuild-compact-with-captions
```

Verify:

```bash
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/captions/frame_index.jsonl
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/captions/shot_index.jsonl
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/indexes/compact_search_index.jsonl
cat /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/reports/shot_caption_report.json
```

Expected:

```text
frame_index.jsonl = 283
shot_index.jsonl = 102
compact_search_index.jsonl = 385
num_caption_empty = 0
num_temporal_caption_empty = 0
```

---

## 14. Suggested implementation order

1. Add `SHOT_INDEX_SCHEMA` in `schemas.py`.
2. Add `ShotCaptionConfig` and `enable_shot_captions` in `config.py`.
3. Add CLI flags in `run_caption_pipeline.py`.
4. Create `shot_captioner.py` with template mode and LLM placeholder fallback.
5. Create `shot_index.py`.
6. Create `shot_caption_report.py`.
7. Add prompt file `prompts/shot_caption_prompt.txt`.
8. Add compact rebuild helper for frame + shot captions.
9. Wire Phase 3 into `run_caption_pipeline.py`.
10. Run local dry-run on `L22_V012`.
11. Check counts and report.
12. Only then run on server.

---

## 15. Notes for Phase 4

Phase 3 output se la input cho Phase 4 TRAKE:

```text
frame_index.jsonl
shot_index.jsonl
shot_evidence.jsonl
-> event_step_index.jsonl
```

Khong nen tron TRAKE vao Phase 3. Phase 3 chi tao shot-level temporal captions va memory.
