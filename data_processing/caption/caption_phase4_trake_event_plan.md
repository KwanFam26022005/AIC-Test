# Caption Phase 4 TRAKE Event-Step Plan

## 0. Phase 3 output gate

Phase 4 chi bat dau sau khi Phase 0/1/2/3 da pass.

Ket qua mong doi tu Phase 3 cho `L22_V012`:

```text
frame_evidence rows       = 283
shot_evidence rows        = 102
frame_index rows          = 283
shot_index rows           = 102
compact_search_index rows = 385
frame docs with caption   = 283
shot docs with caption    = 102
shot caption empty        = 0
temporal caption empty    = 0
duplicate shot ids        = 0
```

Phase 4 se dung:

```text
caption/<video_id>/captions/frame_index.jsonl
caption/<video_id>/captions/shot_index.jsonl
caption/<video_id>/evidence/shot_evidence.jsonl
caption/<video_id>/indexes/compact_search_index.jsonl
```

---

## 1. Muc tieu Phase 4

Phase 4 tao event-step documents cho temporal retrieval/TRAKE, de ho tro query co thu tu:

```text
dau tien ...
sau do ...
cuoi cung ...
```

Phase 4A lam deterministic baseline:

```text
shot_index + shot_evidence
-> event_step_index.jsonl
-> event_step_report.json/md
-> update compact_search_index shot docs with trake_text
```

Trong Phase 4A:

- 1 shot = 1 event-step.
- `step_order` = thu tu shot trong timeline.
- Khong goi LLM.
- Khong tach sub-events trong mot shot.
- Khong dung image/VLM.

Sau khi Phase 4A pass, Phase 4B moi them LLM parser hoac query-aware TRAKE reranker.

---

## 2. Output sau Phase 4

```text
caption/<video_id>/
├── captions/
│   ├── frame_index.jsonl
│   ├── shot_index.jsonl
│   └── event_step_index.jsonl
├── indexes/
│   └── compact_search_index.jsonl
└── reports/
    ├── event_step_report.json
    └── event_step_report.md
```

Expected counts for `L22_V012`:

```text
event_step_index rows          = 102
compact_search_index rows      = 385
frame docs with trake_text     = 0
shot docs with trake_text      = 102
event steps with trake_text    = 102
event steps with empty text    = 0
```

---

## 3. Schema `event_step_index.jsonl`

Moi record:

```json
{
  "schema_version": "caption_event_step_index_v1",
  "document_id": "event_step:L22_V012_shot_0001",
  "doc_type": "event_step",
  "unit_type": "event_step",
  "video_id": "L22_V012",
  "event_id": "L22_V012_event_0001",
  "shot_id": "L22_V012_shot_0001",
  "step_order": 1,
  "start_sec": 0.0,
  "end_sec": 3.0,
  "representative_frame_ids": ["L22_V012_001", "L22_V012_002"],

  "event_caption": "Shot shows ...",
  "current_observation": "The current shot shows ...",
  "before_context": "",
  "after_context": "The next shot covers ...",

  "action_state": "start",
  "temporal_role": "beginning",
  "actors": ["person"],
  "actions": ["speaking"],
  "objects_involved": ["chair"],
  "scene": "city sky sun water",

  "trake_text": "start beginning city water person speaking ...",
  "trake_text_search": "start beginning city water person speaking ...",
  "trake_terms": ["start", "beginning", "..."],

  "source_text": {
    "caption_text": "...",
    "temporal_caption": "...",
    "merged_ocr_text": "...",
    "merged_audio_text": "...",
    "object_text": "...",
    "scene_text": "..."
  },

  "quality": {
    "has_event_caption": true,
    "has_trake_text": true,
    "event_mode": "template",
    "event_model": "",
    "prompt_version": "",
    "fallback_used": false,
    "warnings": []
  }
}
```

Allowed labels:

```text
action_state:
  start | middle | end | transition | result | unknown

temporal_role:
  beginning | continuation | change | completion | unknown
```

---

## 4. Deterministic rules for Phase 4A

### 4.1. Event unit

Use one event step per shot:

```text
shot_index row -> event_step row
```

Rationale:

- Phase 3 already groups temporal context by shot.
- L22_V012 has 102 shots, good granularity for ordered retrieval.
- Avoid noisy per-frame event docs in baseline.

### 4.2. Step order

Sort by:

```text
start_sec, shot_id
```

Then:

```text
step_order = index + 1
event_id = f"{video_id}_event_{step_order:04d}"
document_id = f"event_step:{shot_id}"
```

### 4.3. `action_state`

Heuristic:

```text
if first shot:
    start
elif last shot:
    end
elif scene/object/audio changes strongly from previous:
    transition
elif caption contains result/completion words:
    result
else:
    middle
```

Completion/result keywords can be simple English/Vietnamese hints:

```text
result, finally, completed, ending, conclusion,
ket qua, cuoi cung, hoan thanh, ket thuc
```

### 4.4. `temporal_role`

Heuristic:

```text
if first shot:
    beginning
elif last shot:
    completion
elif major change from previous shot:
    change
else:
    continuation
```

Major change can be:

- Scene tag overlap with previous shot is low.
- Top object labels changed.
- Audio/OCR topic changed.

Phase 4A can implement simple overlap rule first:

```text
scene_overlap_ratio < 0.25 and object_overlap_ratio < 0.25 -> change
```

If evidence sparse, fallback to `continuation`.

### 4.5. Actors/actions/objects

Deterministic baseline:

- `actors`: object labels that look like people:
  - person, man, woman, child, customer, student, player, people
- `objects_involved`: top object labels excluding actors.
- `actions`: keyword hints from captions/audio:
  - speaking, presenting, showing, walking, driving, sitting, standing

If no clear action:

```text
actions = []
action_state can still be middle/start/end
```

Do not hallucinate specific action from weak evidence.

### 4.6. Context

Use neighboring shot captions:

```text
before_context = previous shot temporal_caption or caption_text
current_observation = current shot caption_text
after_context = next shot caption_text
```

Truncate:

```text
before_context max 300 chars
current_observation max 420 chars
after_context max 300 chars
```

### 4.7. `trake_text`

Build for search:

```text
action_state
temporal_role
event_caption
current_observation
actors
actions
objects_involved
scene
important OCR/audio snippets
```

Then normalize:

- remove duplicate whitespace
- cap max chars 900
- no empty text

`trake_text_search`:

- lowercase
- remove accents
- map `đ -> d`

---

## 5. Code layout

Add:

```text
caption_pipeline/
├── trake_event_builder.py
├── event_step_index.py
├── event_step_report.py
└── prompts/
    └── trake_event_step_prompt.txt
```

Modify:

```text
caption_pipeline/config.py
caption_pipeline/schemas.py
caption_pipeline/shot_index.py
run_caption_pipeline.py
```

---

## 6. Config and CLI

### 6.1. `config.py`

Add:

```python
@dataclass
class TrakeEventConfig:
    event_mode: str = "template"  # "template" or "llm"
    max_before_context_chars: int = 300
    max_current_observation_chars: int = 420
    max_after_context_chars: int = 300
    max_trake_text_chars: int = 900
```

Add to `PipelineConfig`:

```python
enable_trake_events: bool = False
trake_event: TrakeEventConfig = field(default_factory=TrakeEventConfig)
```

### 6.2. CLI

Add:

```bash
--enable-trake-events
--trake-event-mode template
```

Dependency:

```text
Phase 4 requires --enable-shot-captions
Phase 4 indirectly requires --enable-frame-captions
```

If user runs:

```bash
--enable-trake-events
```

without:

```bash
--enable-shot-captions
```

return a clear CLI error.

---

## 7. `trake_event_builder.py`

API:

```python
def build_trake_event_steps(
    shot_index: list[dict],
    shot_evidence: list[dict],
    cfg: PipelineConfig,
) -> list[dict]:
    ...
```

Output intermediate event record:

```json
{
  "shot_id": "L22_V012_shot_0001",
  "event_id": "L22_V012_event_0001",
  "step_order": 1,
  "event_caption": "...",
  "current_observation": "...",
  "before_context": "",
  "after_context": "...",
  "action_state": "start",
  "temporal_role": "beginning",
  "actors": [],
  "actions": [],
  "objects_involved": [],
  "scene": "...",
  "trake_text": "...",
  "event_mode": "template",
  "fallback_used": false,
  "warnings": []
}
```

LLM placeholder:

If `--trake-event-mode llm` is selected but not implemented:

```text
event_mode = fallback
fallback_used = true
warnings += ["llm_mode_not_implemented_using_template"]
```

---

## 8. `event_step_index.py`

API:

```python
def build_event_step_index(
    shot_index: list[dict],
    shot_evidence: list[dict],
    event_records: list[dict],
) -> list[dict]:
    ...
```

Responsibilities:

- Build `caption_event_step_index_v1`.
- Preserve timing and representative frames from shot index.
- Add `trake_text_search`.
- Add `trake_terms`.
- Add `source_text`.
- Add `quality`.

Also add compact rebuild helper:

```python
def rebuild_compact_with_trake_text(
    compact_docs: list[dict],
    event_records: list[dict],
) -> list[dict]:
    ...
```

Update rules:

- `unit_type == "shot"`:
  - lookup by `unit_id == shot_id`
  - set `trake_text`
  - rebuild `all_text`
- `unit_type == "frame"`:
  - leave `trake_text` empty in Phase 4A

---

## 9. `event_step_report.py`

Report JSON:

```json
{
  "video_id": "L22_V012",
  "created_at": "...",
  "num_event_steps": 102,
  "num_event_caption_generated": 102,
  "num_trake_text_generated": 102,
  "num_trake_text_empty": 0,
  "num_fallback_used": 0,
  "event_mode_counts": {
    "template": 102
  },
  "action_state_counts": {
    "start": 1,
    "middle": 100,
    "end": 1
  },
  "temporal_role_counts": {
    "beginning": 1,
    "continuation": 100,
    "completion": 1
  },
  "duplicate_document_ids": 0,
  "duplicate_event_ids": 0,
  "duplicate_shot_ids": 0,
  "num_event_warnings": 0,
  "event_warnings": [],
  "summary_warnings": []
}
```

Markdown:

- ASCII-safe.
- Use `[PASS]` and `[FAIL]`.
- Cap warnings at first 50.

---

## 10. `run_caption_pipeline.py` flow

Target flow:

```text
Phase 0: evidence alignment
Phase 1: shot grouping + compact baseline
Phase 2: frame captions
Phase 3: shot captions
Phase 4: TRAKE event steps
Final compact rebuild:
  frame caption_text
  shot caption_text
  shot trake_text
```

Minimal-risk implementation:

- Keep current Phase 2/3 compact rebuild behavior.
- After Phase 4, rebuild compact again with `trake_text`.
- Later can refactor to write compact once.

Acceptance problems should include:

```text
empty event captions
empty trake_text
duplicate event IDs
```

---

## 11. Prompt placeholder for Phase 4B

File:

```text
caption_pipeline/prompts/trake_event_step_prompt.txt
```

Content:

```text
You are an event-step extraction module for temporal video retrieval.

Given shot caption, temporal caption, neighboring context, objects, OCR,
and audio, create one event-step JSON object.

Rules:
- Use only the provided evidence.
- Do not invent actions if evidence is weak.
- If no clear action appears, use actions=[] and action_state="unknown".
- Keep trake_text concise and search-friendly.
- Output valid JSON only.

Expected JSON:
{
  "event_caption": "...",
  "action_state": "start|middle|end|transition|result|unknown",
  "temporal_role": "beginning|continuation|change|completion|unknown",
  "actors": ["..."],
  "actions": ["..."],
  "objects_involved": ["..."],
  "scene": "...",
  "before_context": "...",
  "current_observation": "...",
  "after_context": "...",
  "trake_text": "..."
}
```

Do not call this prompt in Phase 4A.

---

## 12. Acceptance checklist

Phase 4 pass when:

```text
event_step_index rows = 102
event caption empty = 0
trake_text empty = 0
duplicate document ids = 0
duplicate event ids = 0
duplicate shot ids = 0
compact_search_index rows = 385
frame docs with caption_text = 283
shot docs with caption_text = 102
shot docs with trake_text = 102
frame docs with trake_text = 0
```

Quality checks:

```text
trake_text_search has no Vietnamese "đ"
no double period ".." in event_caption/current_observation/trake_text
max trake_text length <= 900
num_event_warnings = 0 for template run
```

---

## 13. Server command after coding

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
  --enable-trake-events \
  --trake-event-mode template \
  --rebuild-compact-with-captions
```

Verify:

```bash
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/captions/event_step_index.jsonl
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/indexes/compact_search_index.jsonl
cat /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/reports/event_step_report.json
```

Expected:

```text
event_step_index.jsonl = 102
compact_search_index.jsonl = 385
num_trake_text_empty = 0
```

---

## 14. Suggested implementation order

1. Add `EVENT_STEP_INDEX_SCHEMA` in `schemas.py`.
2. Add `TrakeEventConfig` and `enable_trake_events` in `config.py`.
3. Add CLI flags in `run_caption_pipeline.py`.
4. Create `trake_event_builder.py`.
5. Create `event_step_index.py`.
6. Create `event_step_report.py`.
7. Add `prompts/trake_event_step_prompt.txt`.
8. Wire Phase 4 into `run_caption_pipeline.py`.
9. Rebuild compact index with `trake_text`.
10. Run dry-run on `L22_V012`.
11. Check counts/report/quality.
12. Only then run on server.

---

## 15. Notes for retrieval

After Phase 4, retrieval can use:

```text
frame docs:
  frame caption, OCR, audio, object, scene

shot docs:
  shot caption, temporal caption, trake_text, OCR, audio, object, scene

event_step docs:
  trake_text, action_state, temporal_role, before/current/after context
```

For TRAKE-style queries, search should prioritize:

```text
event_step_index.trake_text
event_step_index.current_observation
event_step_index.temporal_role
event_step_index.action_state
```

Then enforce timestamp ordering across matched event steps.
