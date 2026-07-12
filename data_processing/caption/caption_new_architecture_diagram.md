# Caption Pipeline - New Architecture Diagram

Tài liệu này ghi lại hướng kiến trúc caption mới sau các thí nghiệm với OCR grounding và visual frame caption trên `L22_V012`.

## 1. Kết luận chính

Sau khi test:

- OCR với `full frame + tọa độ bbox` không ổn định cho Vintern-3B-beta.
- OCR crop/ROI cho kết quả tốt hơn rõ rệt.
- Visual frame caption bằng Qwen2.5-VL-7B-Instruct tốt hơn Vintern-3B-beta trên `L22_V012`.
- Vintern-3B-beta có xu hướng nghiêng sang OCR/text-only khi dùng cho visual caption.
- OCR/object nếu đưa thẳng vào caption có thể gây noise.

Vì vậy pipeline mới nên tách rõ:

```text
OCR branch      -> search text evidence
Caption branch  -> visual semantic evidence
Audio branch    -> speech/context evidence
```

Ba nhánh chỉ nên hội tụ ở tầng `Shot/Event/TRAKE` bằng text LLM.

## 2. Diagram tổng thể

```text
                         ┌────────────────────┐
                         │   Keyframe Images   │
                         └─────────┬──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
        ┌─────────────────────┐       ┌────────────────────────┐
        │ OCR Branch           │       │ Visual Caption Branch  │
        └─────────┬───────────┘       └───────────┬────────────┘
                  │                               │
                  ▼                               ▼
        ┌─────────────────────┐       ┌────────────────────────┐
        │ PP-OCR Detection     │       │ Frame Selector          │
        │ boxes only           │       │ all/representative      │
        └─────────┬───────────┘       └───────────┬────────────┘
                  │                               │
                  ▼                               ▼
        ┌─────────────────────┐       ┌────────────────────────┐
        │ Expanded Crop / ROI  │       │ Full Frame Image        │
        │ from bbox            │       │ no OCR boxes, no object │
        └─────────┬───────────┘       └───────────┬────────────┘
                  │                               │
                  ▼                               ▼
        ┌─────────────────────┐       ┌────────────────────────┐
        │ VietOCR Fast Pass    │       │ Qwen2.5-VL Frame Caption│
        │ text + rec_conf      │       │ visual caption only     │
        └─────────┬───────────┘       └───────────┬────────────┘
                  │                               │
                  ▼                               ▼
        ┌─────────────────────┐       ┌────────────────────────┐
        │ Confidence Gate      │       │ Caption Validator       │
        │ high / low quality   │       │ length, repetition, JSON│
        └──────┬────────┬─────┘       └───────────┬────────────┘
               │        │                         │
      high conf│        │low conf                 │pass
               ▼        ▼                         ▼
   ┌────────────────┐  ┌───────────────────┐  ┌──────────────────────┐
   │ Accept VietOCR │  │ Vintern Crop OCR   │  │ frame_caption.jsonl   │
   │ no VLM cost    │  │ crop/ROI only      │  │ visual-only caption   │
   └───────┬────────┘  └─────────┬─────────┘  └───────────┬──────────┘
           │                     │                        │
           └──────────┬──────────┘                        │
                      ▼                                   │
          ┌────────────────────────┐                      │
          │ OCR Validator / Cleaner│                      │
          │ normalize, reject noise│                      │
          └───────────┬────────────┘                      │
                      ▼                                   │
          ┌────────────────────────┐                      │
          │ ocr_index.jsonl         │                      │
          │ searchable text only    │                      │
          └───────────┬────────────┘                      │
                      │                                   │
                      └──────────────┬────────────────────┘
                                     ▼
                          ┌────────────────────┐
                          │ Audio / ASR Branch │
                          │ transcript by time │
                          └─────────┬──────────┘
                                    ▼
              ┌────────────────────────────────────────┐
              │ Text LLM Fusion                         │
              │ Qwen2.5-7B-Instruct                     │
              │ inputs: frame captions + ASR + shot map │
              │ optional: short OCR summary only if need │
              └───────────────────┬────────────────────┘
                                  ▼
                 ┌────────────────────────────────┐
                 │ Shot Caption / Event / TRAKE   │
                 │ final search-ready semantics   │
                 └────────────────────────────────┘
```

## 3. Visual frame caption branch

Visual frame caption phải trả lời câu hỏi:

```text
Frame này đang nhìn thấy cảnh gì?
```

Không nên bắt frame caption làm OCR chính xác, không nên nhét object list, và không nên nhét ASR vào prompt ở bước này.

### Input

```text
full keyframe image
```

### Model chính

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

Kết quả thí nghiệm trên `L22_V012`:

| Model | Frames | Avg time/frame | Warnings | Nhận xét |
|---|---:|---:|---:|---|
| Vintern-3B-beta | 283 | ~2.325s | 7 | Hay bị text-only/OCR-first |
| Qwen2.5-VL-7B | 283 | ~1.971s | 0 | Caption giàu visual hơn |

Vì vậy Qwen2.5-VL nên là default cho visual frame caption.

### Prompt định hướng

```text
Describe this video keyframe for retrieval in concise English.
Mention only clearly visible scene, people, objects, actions, layout, and prominent readable text.
Do not guess identities, locations, causes, or events outside the image.
Return plain caption text only, under 320 characters.
```

### Output

```json
{
  "video_id": "L22_V012",
  "frame_id": "L22_V012_216",
  "image_path": ".../216.jpg",
  "provider": "qwen",
  "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
  "caption": "Blue street signs read ... Several people wearing helmets stand nearby.",
  "elapsed_seconds": 1.97,
  "warning": ""
}
```

## 4. OCR branch

OCR branch phải trả lời câu hỏi:

```text
Trong frame có text nào cần index/search?
```

OCR không nên dùng làm input trực tiếp cho frame caption. OCR nên là một evidence riêng.

### Input

```text
PP-OCR bbox -> expanded crop / ROI
```

### Flow

```text
PP-OCR detect
  -> expanded crop
  -> VietOCR fast pass
  -> confidence/text-quality gate
      -> high confidence: accept VietOCR
      -> low confidence: Vintern crop fallback
  -> OCR validator/cleaner
  -> OCR index
```

### Lý do giữ VietOCR

- VietOCR nhanh và rẻ.
- VietOCR cung cấp `rec_conf` để gating.
- Nếu bỏ VietOCR và gửi toàn bộ bbox qua Vintern, latency tăng mạnh.
- Vintern có thể sửa tốt crop khó, nhưng output format chưa ổn định.

### Không khuyến nghị

```text
full frame + bbox coordinate -> Vintern OCR
```

Thí nghiệm grounding cho thấy Vintern-3B-beta có thể đọc nhầm vùng chữ khác khi chỉ truyền tọa độ.

## 5. Audio branch

Audio branch giữ vai trò cung cấp speech/context theo thời gian.

```text
audio -> ASR -> transcript segments -> align by timestamp/shot
```

Audio không đi vào visual frame caption stage. Audio chỉ hội tụ ở tầng shot/event.

## 6. Fusion ở Shot/Event/TRAKE

Fusion nên xảy ra muộn:

```text
frame visual captions
+ ASR transcript around shot
+ shot boundary / timestamp
+ optional short OCR summary
-> Qwen2.5-7B-Instruct text LLM
-> shot caption / event / TRAKE text
```

Không nên đưa toàn bộ raw OCR vào text LLM nếu không cần. Chỉ đưa OCR summary ngắn khi:

- shot có text quan trọng,
- visual caption không đủ,
- query/search cần text evidence.

## 7. Rút gọn production diagram

```text
Keyframes
├── OCR
│   └── PP-OCR -> crop -> VietOCR -> gate -> Vintern crop fallback -> OCR index
│
├── Visual Caption
│   └── full frame -> Qwen2.5-VL -> frame caption index
│
└── Audio
    └── ASR -> transcript segments

OCR index + frame captions + ASR
    -> Qwen2.5 text LLM
    -> Shot caption / Event / TRAKE
    -> Final searchable index
```

## 8. Quyết định hiện tại

| Thành phần | Model/chiến lược đề xuất | Lý do |
|---|---|---|
| OCR detect | PP-OCR | Lấy bbox ổn định, nhanh |
| OCR fast pass | VietOCR | Rẻ, có confidence để gate |
| OCR fallback | Vintern-3B-beta on crop/ROI | Sửa tốt crop khó, tránh coordinate grounding |
| Frame caption | Qwen2.5-VL-7B-Instruct | Caption visual tốt hơn Vintern trên test hiện tại |
| Shot/Event/TRAKE | Qwen2.5-7B-Instruct text LLM | Fuse caption + ASR + optional OCR summary |

## 9. Nguyên tắc quan trọng

1. OCR và caption không trộn sớm.
2. OCR dùng crop/ROI, không dùng full-frame coordinate làm path chính.
3. Frame caption dùng full frame, không dùng raw OCR boxes.
4. Object detection không đưa thẳng vào caption nếu chưa có bằng chứng giúp cải thiện.
5. Fusion chỉ làm ở shot/event level bằng text LLM.
6. OCR là search text evidence; visual caption là semantic evidence; ASR là speech evidence.

