# Caption Phase 2 Frame Fuser Plan

## 0. Phase 0/1 output gate

Da kiem tra output moi:

```text
caption/L22_V012/
├── evidence/frame_evidence.jsonl
├── evidence/shot_evidence.jsonl
├── indexes/compact_search_index.jsonl
└── reports/evidence_alignment_report.json
```

Ket qua validation:

```text
frame_evidence rows        = 283
shot_evidence rows         = 102
compact_search_index rows  = 385
frame docs                 = 283
shot docs                  = 102
timestamp mismatch         = 0
missing timestamp          = 0
empty frame_search_text    = 0
empty compact all_text     = 0
duplicate frame ids        = 0
duplicate compact unit ids = 0
object/scene count overlap = 0
audio features total       = 226
audio features usable      = 217
scene labels removed       = 5
```

Ket luan: Phase 0/1 dat chuan de tiep tuc Phase 2.

Luu y chat luong:

- 20/283 frames khong co OCR text sach.
- 17/283 frames khong co countable object.
- 3/283 frames khong co audio window.
- Day khong phai loi blocking, vi moi frame van co `frame_search_text` va co it nhat mot nguon evidence de search.

---

## 1. Muc tieu Phase 2

Phase 2 tao caption frame-level tu evidence da align, khong detect lai object, khong chay lai OCR/audio.

Input chinh:

```text
caption/<video_id>/evidence/frame_evidence.jsonl
caption/<video_id>/evidence/shot_evidence.jsonl
caption/<video_id>/indexes/compact_search_index.jsonl
```

Output moi:

```text
caption/<video_id>/captions/frame_index.jsonl
caption/<video_id>/reports/frame_caption_report.json
caption/<video_id>/reports/frame_caption_report.md
```

Output co the cap nhat:

```text
caption/<video_id>/indexes/compact_search_index.jsonl
```

Trong Phase 2, `compact_search_index.jsonl` nen duoc rebuild them `caption_text`, nhung van giu cac field cu de khong pha schema ES hien tai.

---

## 2. Nguyen tac thiet ke

### 2.1. Lam baseline deterministic truoc

Truoc khi goi LLM, can co che do deterministic/template:

```text
frame_evidence -> normalized evidence summary -> template caption -> frame_index.jsonl
```

Ly do:

- Chay nhanh de test schema va ES ingestion.
- Khong phu thuoc GPU/model.
- Lam fallback neu LLM loi hoac output khong parse duoc JSON.
- Tao baseline de so sanh chat luong khi them LLM.

### 2.2. Text-only fuser, khong dung anh truc tiep trong Phase 2

Phase 2 chi doc evidence text:

- OCR text
- audio text quanh frame
- object counts
- scene tags
- RAM tags

Khong dung image path de goi VLM trong phase nay. Neu can VLM/image captioning, de Phase 2.5 hoac phase rieng sau khi text-only fuser on dinh.

### 2.3. Schema additive

Khong sua/y xoa schema Phase 0/1. Them output moi va neu rebuild compact index thi chi dien cac field dang de trong:

```json
{
  "caption_text": "...",
  "trake_text": ""
}
```

---

## 3. Schema `frame_index.jsonl`

Moi record:

```json
{
  "schema_version": "caption_frame_index_v1",
  "document_id": "caption_frame:L22_V012_001",
  "unit_type": "frame",
  "video_id": "L22_V012",
  "frame_id": "L22_V012_001",
  "canonical_frame_id": "L22_V012_001",
  "timestamp_sec": 0.0,
  "image_relpath": "L22_V012/001.jpg",

  "caption_text": "Khung hinh hien thi ...",
  "caption_text_search": "khung hinh hien thi ...",
  "caption_terms": ["..."],

  "evidence_text": {
    "ocr_text": "giay",
    "audio_text": "Chao mung quy vi ...",
    "object_text": "2 person 1 chair",
    "scene_text": "city sky sun water"
  },

  "search_fields": {
    "all_text": "ocr audio object scene caption",
    "caption_boost_text": "caption repeated or concise terms",
    "visual_text": "object scene ram tags",
    "spoken_text": "audio",
    "onscreen_text": "ocr"
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
    "fallback_used": false,
    "warnings": []
  }
}
```

Ghi chu:

- `caption_text` la field chinh cho search.
- `caption_text_search` la ban normalize de search/analyzer.
- `search_fields.all_text` phuc vu ES ingest.
- `quality.caption_mode` co the la `template`, `llm`, hoac `fallback`.

---

## 4. Code layout can them

```text
data_processing/caption/
├── run_caption_pipeline.py
└── caption_pipeline/
    ├── frame_caption_fuser.py
    ├── caption_index.py
    ├── caption_report.py
    └── prompts/
        └── frame_caption_prompt.txt
```

### 4.1. `frame_caption_fuser.py`

Trach nhiem:

- Doc tung `frame_evidence`.
- Chon evidence tot nhat theo thu tu:
  - OCR: `ocr_evidence.ocr_text`
  - audio: `audio_evidence.audio_text`
  - object: `object_evidence.important_objects` hoac `object_counts`
  - scene: `scene_tags`
  - RAM tags: chi lay tag ngan, tranh spam tag dai.
- Tao `caption_text` bang template mode.
- Neu bat `--caption-mode llm`, tao prompt va parse JSON output.
- Neu LLM loi, fallback ve template caption.

### 4.2. `caption_index.py`

Trach nhiem:

- Build `caption_frame_index_v1`.
- Rebuild compact docs voi `caption_text` cho frame docs.
- Giu nguyen shot docs trong Phase 2, chua dien `shot caption`.

### 4.3. `caption_report.py`

Trach nhiem:

- Dem `num_frames`.
- Dem `num_caption_generated`.
- Dem `num_caption_empty`.
- Dem `num_fallback_used`.
- Dem coverage theo evidence: OCR/audio/object/scene.
- Liet ke warning/failure theo frame.

---

## 5. Template caption rules

Template mode nen tao cau ngan, search-friendly, khong qua van hoa:

```text
[scene phrase]. [object phrase]. [ocr phrase]. [audio phrase].
```

Rules:

- Neu co OCR: them `On-screen text: ...`.
- Neu co audio: them `Spoken content: ...`.
- Neu co object counts: them `Visible objects: ...`.
- Neu chi co scene/RAM tags: tao caption mo ta boi canh.
- Truncate audio/OCR de tranh caption qua dai:
  - OCR max 160 chars
  - audio max 220 chars
  - object phrase max 12 labels
  - RAM tags max 10 tags

Vi du:

```text
Scene shows city, sky, sun and water. On-screen text: "giay". Spoken content mentions the 60 giay program from HTV.
```

Khong nen:

- Lap lai toan bo ASR dai.
- Dua bbox/confidence vao caption.
- Gop OCR review text vao primary caption.
- Copy nguyen `frame_search_text` lam caption.

---

## 6. CLI thay doi

Them cac option vao `run_caption_pipeline.py`:

```bash
python run_caption_pipeline.py \
  --video-id L22_V012 \
  --ocr-jsonl ... \
  --object-jsonl ... \
  --audio-features ... \
  --output-dir caption \
  --frame-window-before 5 \
  --frame-window-after 5 \
  --shot-max-gap-sec 5 \
  --shot-max-duration-sec 20 \
  --enable-frame-captions \
  --caption-mode template \
  --rebuild-compact-with-captions
```

Mac dinh:

```text
--enable-frame-captions false
--caption-mode template
--rebuild-compact-with-captions false
```

Neu chua muon anh huong Phase 0/1, co the them script rieng:

```text
run_frame_caption_fuser.py
```

Nhung khuyen nghi gan vao `run_caption_pipeline.py` bang flag de sau nay batch de hon.

---

## 7. Output sau Phase 2

```text
caption/L22_V012/
├── evidence/
│   ├── frame_evidence.jsonl
│   └── shot_evidence.jsonl
├── captions/
│   └── frame_index.jsonl
├── indexes/
│   └── compact_search_index.jsonl
└── reports/
    ├── evidence_alignment_report.json
    ├── evidence_alignment_report.md
    ├── frame_caption_report.json
    └── frame_caption_report.md
```

Expected counts for L22_V012:

```text
frame_index rows = 283
caption generated = 283
caption empty = 0
compact_search_index rows = 385
frame docs with caption_text = 283
shot docs with caption_text = 0
```

Shot docs se duoc dien caption o Phase 3.

---

## 8. Acceptance checklist

Phase 2 dat chuan khi:

- `frame_index.jsonl` co 283 rows cho L22_V012.
- Moi row co `caption_text` khong rong.
- Moi row co `caption_text_search` khong rong.
- `document_id` va `canonical_frame_id` khong duplicate.
- `timestamp_sec` va `image_relpath` duoc preserve tu frame evidence.
- `quality.caption_mode` co gia tri hop le.
- Neu `--rebuild-compact-with-captions`, compact index van co 385 rows.
- Compact frame docs co `caption_text`; shot docs chua caption van khong loi.
- Report ghi ro so fallback/warnings.
- Chay lai pipeline nhieu lan cho cung input cho output deterministic neu dung `template`.

---

## 9. Phase 3 sau khi Phase 2 pass

Sau khi frame captions on dinh, Phase 3 moi build ReCap shot captioner:

```text
shot_evidence
+ frame_index captions trong shot
+ previous shot memory
-> shot_index.jsonl
```

Phase 3 output:

```text
caption/<video_id>/captions/shot_index.jsonl
```

Tam thoi khong viet code Phase 3 truoc khi Phase 2 co report sach.
