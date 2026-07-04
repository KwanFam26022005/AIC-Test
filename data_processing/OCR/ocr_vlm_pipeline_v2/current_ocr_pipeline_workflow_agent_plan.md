# Workflow hiện tại: PP-OCRv6 DET → VietOCR Wordlist Gating → Vintern Line/Group Fallback

Tài liệu này mô tả chi tiết workflow hiện tại để agent có thể refactor/coding lại thành pipeline rõ ràng, có thể chạy trên 1 frame trước, sau đó mở rộng sang nhiều frame/video.

Notebook nguồn hiện tại:

```text
ppocrv6_det_group_line_vietocr_vintern_wordlist_gating_v2.ipynb
```

Mục tiêu chính của workflow:

```text
1 frame ảnh
→ detect text line bằng PP-OCRv6_medium_det
→ crop từng line bằng perspective
→ group các line-box gần/chồng nhau để giữ context
→ OCR line bằng VietOCR
→ đánh giá độ tin cậy bằng wordlist + composite score
→ lọc logo/timestamp/60 giây/noise
→ nếu line đáng nghi: gọi Vintern line fallback
→ nếu group nhiều dòng đáng nghi: gọi Vintern group fallback
→ chỉ đưa text đã accept vào group_text_clean / final_clean_text
→ đưa text chưa chắc đúng vào group_text_review / final_review_text
→ export CSV/JSON/TXT/visualization
```

---

## 1. Vấn đề pipeline đang giải quyết

### 1.1. Không thể gửi tất cả crop sang VLM

Với quy mô lớn:

```text
~1M frame × ~10 dòng/frame ≈ 10M line crops
```

Nếu mọi line đều gửi sang Vintern/VLM thì chi phí GPU rất cao. Vì vậy pipeline cần một tầng **gating** để phân loại:

```text
auto_accept       → đủ tin cậy, không gửi VLM
vlm_candidate     → nghi ngờ, cần line/group Vintern
structural_filter → logo/time/counter/noise, bỏ qua
review_only       → chưa đủ chắc để index
```

### 1.2. VietOCR đọc khá tốt tiếng Việt nhưng không luôn có confidence

Trong notebook hiện tại, VietOCR được gọi với:

```python
vietocr_predictor.predict(crop, return_prob=True)
```

Tuy nhiên tùy version/config, output có thể là:

```text
(text, prob)
[text, prob]
text only
return_prob_missing
return_prob_no_conf
```

Nếu không lấy được confidence thật, pipeline dùng fallback:

```python
rec_conf = 0.50
```

và bật logic riêng cho trường hợp `rec_conf_flat/missing`.

### 1.3. Text review không được index

Trước đây có lỗi nguy hiểm:

```text
group_text = clean_text if clean_text else review_text
```

Điều này khiến text chưa được accept vẫn lọt vào index. Bản hiện tại sửa thành:

```text
group_text_clean  → chỉ chứa text đã accept
group_text_review → chứa text nghi ngờ để debug/manual review
group_text        → bằng group_text_clean, KHÔNG fallback sang review
```

---

## 2. Kiến trúc tổng quan

```text
┌──────────────────────────┐
│ Input frame image         │
└─────────────┬────────────┘
              │
              v
┌──────────────────────────┐
│ PP-OCRv6_medium_det       │
│ Output: line polygons     │
└─────────────┬────────────┘
              │
              v
┌──────────────────────────┐
│ Validate / filter boxes   │
│ min_width, min_height     │
│ aspect ratio, det_score   │
└─────────────┬────────────┘
              │
              v
┌──────────────────────────┐
│ Perspective crop line     │
│ Save line crop            │
└─────────────┬────────────┘
              │
              v
┌──────────────────────────┐
│ Group nearby/stacked lines│
│ Save group crop context   │
└─────────────┬────────────┘
              │
              v
┌──────────────────────────┐
│ VietOCR line recognition  │
│ Try return_prob=True      │
└─────────────┬────────────┘
              │
              v
┌──────────────────────────┐
│ Wordlist + feature score  │
│ lex_ratio, diacritic_susp │
│ charset_penalty, det_score│
│ composite_score           │
└─────────────┬────────────┘
              │
              v
┌──────────────────────────┐
│ Gating decision           │
│ auto_accept / VLM / filter│
└──────┬──────────────┬─────┘
       │              │
       │              v
       │       ┌────────────────────┐
       │       │ Vintern line OCR    │
       │       │ if line suspicious  │
       │       └──────────┬─────────┘
       │                  │
       v                  v
┌──────────────────────────────────┐
│ Update line final_text/source     │
│ keep_for_index / need_review      │
└────────────────┬─────────────────┘
                 │
                 v
┌──────────────────────────────────┐
│ Vintern group OCR                 │
│ if multi-line group suspicious    │
└────────────────┬─────────────────┘
                 │
                 v
┌──────────────────────────────────┐
│ group_text_clean / review         │
│ final_clean_text / review_text    │
└────────────────┬─────────────────┘
                 │
                 v
┌──────────────────────────────────┐
│ Export CSV / JSON / TXT / Image   │
└──────────────────────────────────┘
```

---

## 3. Modules agent nên tách ra

Agent nên refactor notebook thành các module/function riêng:

```text
ocr_pipeline/
├── config.py
├── detector.py
├── cropping.py
├── grouping.py
├── recognizers/
│   ├── vietocr_recognizer.py
│   └── vintern_recognizer.py
├── scoring/
│   ├── wordlist.py
│   ├── features.py
│   └── gating.py
├── outputs.py
└── run_single_frame.py
```

Hoặc nếu giữ trong notebook, vẫn nên tổ chức theo các cell tương ứng.

---

## 4. Cấu hình chính

### 4.1. Detector config

```python
CONFIG = {
    "det_model_name": "PP-OCRv6_medium_det",
    "det_limit_type": "min",
    "det_limit_side_len": 960,
    "det_thresh": 0.30,
    "det_box_thresh": 0.50,
    "det_unclip_ratio": 1.8,
}
```

Ý nghĩa:

| Tham số | Vai trò |
|---|---|
| `det_limit_side_len` | Resize ảnh đầu vào cho detector. Tăng lên 1216 nếu miss chữ nhỏ. |
| `det_thresh` | Ngưỡng pixel text confidence. Giảm nếu miss text. |
| `det_box_thresh` | Ngưỡng box confidence. Giảm nếu miss box, tăng nếu quá nhiều noise. |
| `det_unclip_ratio` | Nới box text. Tăng nếu crop bị cắt dấu/chữ. |

### 4.2. Line box filtering

```python
"drop_low_det_score_below": 0.0,
"min_box_width": 10,
"min_box_height": 8,
"min_box_aspect_ratio": 0.25,
```

Mục tiêu: loại box quá nhỏ/không hợp lệ trước khi crop/OCR.

### 4.3. Perspective crop config

```python
"perspective_padding": 12,
"crop_min_height_for_ocr": 72,
"crop_upscale_max_factor": 3.5,
"add_white_border": 10,
"contrast_factor": 1.25,
```

Mục tiêu: chuẩn hóa line crop cho VietOCR/Vintern.

Luồng xử lý:

```text
polygon box
→ order points
→ expand polygon bằng padding
→ perspective transform
→ upscale nếu height thấp
→ tăng contrast
→ thêm white border
→ save crop
```

### 4.4. Grouping config

```python
"group_min_x_overlap_ratio": 0.12,
"group_max_vertical_gap_ratio": 1.80,
"group_min_lines": 2,
"group_crop_padding": 28,
```

Mục tiêu: group các dòng text thuộc cùng block/context.

Ví dụ:

```text
ĐƯỜNG
VÕ VĂN KIỆT
```

Hai line này nên nằm cùng group để Vintern group-crop có đủ context.

### 4.5. VietOCR config

```python
"vietocr_config": "vgg_transformer",
"vietocr_beamsearch": True,
"use_vietocr_return_prob": True,
"rec_conf_fallback_when_missing": 0.50,
```

Nếu `return_prob=True` không hoạt động, pipeline sẽ gán:

```python
rec_conf = 0.50
rec_conf_source = "return_prob_missing" hoặc "text_only"
```

Sau đó dùng rule riêng cho trường hợp confidence missing/flat.

### 4.6. Wordlist config

```python
"wordlist_paths": [
    "/content/drive/MyDrive/vietnamese/vn_dictionary.txt",
    "/content/drive/MyDrive/vietnamese/general_dict.txt",
    "/content/vn_dictionary.txt",
    "/content/general_dict.txt",
],
"wordlist_min_entries_to_enable": 1000,
"neutral_lex_ratio_if_no_wordlist": 0.50,
```

`vn_dictionary.txt` và `general_dict.txt` dùng để tạo:

```python
VN_WORDSET
VN_BASE_TO_VARIANTS
```

Trong đó:

```text
VN_WORDSET:
  tập token hợp lệ

VN_BASE_TO_VARIANTS:
  token không dấu → các biến thể có dấu hợp lệ
```

Ví dụ:

```text
"van" → {"văn", "vân", "vạn", ...}
"kiet" → {"kiệt", ...}
```

---

## 5. Data structures

### 5.1. `line_item`

Mỗi detected line nên lưu dạng dict:

```python
line_item = {
    "line_id": int,
    "det_idx": int,
    "box": [[x, y], [x, y], [x, y], [x, y]],
    "bbox_xyxy": [x1, y1, x2, y2],
    "det_score": float,

    "persp_crop_path": str,
    "group_id": int | None,
    "group_crop_path": str | None,

    "vietocr_text": str,
    "rec_conf": float,
    "rec_conf_source": str,
    "rec_conf_flat_run": bool,

    "lex_ratio": float,
    "diacritic_susp": float,
    "charset_penalty": float,
    "repetition_penalty": float,
    "det_score_norm": float,
    "composite_score": float,
    "quality_score": float,
    "priority": float,

    "tokens": list[str],
    "eval_tokens": list[str],
    "diacritic_suspicious_tokens": list[str],
    "oov_tokens": list[str],

    "filter_status": str,
    "send_to_vintern": bool,
    "is_filtered": bool,

    "vintern_text": str | None,
    "vintern_composite_score": float | None,
    "agreement_similarity": float | None,

    "final_text": str,
    "final_source": str,
    "keep_for_index": bool,
    "need_review": bool,

    "group_text": str,
    "group_text_review": str,
}
```

### 5.2. `group`

Mỗi group chứa nhiều line:

```python
group = {
    "group_id": int,
    "reading_order": int,
    "line_indices": list[int],
    "num_lines": int,
    "is_multiline_group": bool,

    "bbox_xyxy": [x1, y1, x2, y2],
    "group_crop_path": str,

    "group_text_clean": str,
    "group_text_review": str,
    "group_text": str,
    "group_text_lines": list[str],
    "group_review_lines": list[str],

    "num_keep_lines": int,
    "num_filtered_lines": int,
    "num_vlm_candidates": int,
    "num_vintern_lines": int,

    "mean_det_score": float,
    "mean_composite_score": float,
    "mean_rec_conf": float,
    "group_vlm_priority": float,

    "group_vintern_text": str | None,
    "group_vintern_composite_score": float | None,
    "group_vintern_agreement_similarity": float | None,
    "group_final_source": str | None,
    "group_keep_for_index": bool | None,

    "need_review": bool,
}
```

---

## 6. Detection stage

### 6.1. Input

```python
IMAGE_PATH = "/content/frame.jpg"
img_rgb = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]
```

### 6.2. Detector

```python
detector = TextDetection(
    model_name="PP-OCRv6_medium_det",
    device=PADDLE_DEVICE,
    engine="paddle_static",
    limit_side_len=CONFIG["det_limit_side_len"],
    limit_type=CONFIG["det_limit_type"],
    thresh=CONFIG["det_thresh"],
    box_thresh=CONFIG["det_box_thresh"],
    unclip_ratio=CONFIG["det_unclip_ratio"],
    enable_mkldnn=False,
    cpu_threads=4,
)
```

### 6.3. Output parse

Detector output cần parse ra:

```python
boxes: list[np.ndarray]  # shape [4, 2]
det_scores: list[float]
```

Agent cần implement function:

```python
def parse_text_detection_output(det_output) -> tuple[list[np.ndarray], list[float]]:
    ...
```

Nên hỗ trợ các field:

```text
dt_polys
rec_polys
boxes
dt_scores
scores
```

---

## 7. Cropping stage

### 7.1. Line crop

Mỗi polygon box được crop bằng perspective transform:

```python
line_crop = crop_perspective_line(img_rgb, box, pad=CONFIG["perspective_padding"])
line_crop = prepare_crop_for_ocr(
    line_crop,
    min_height=CONFIG["crop_min_height_for_ocr"],
    max_upscale_factor=CONFIG["crop_upscale_max_factor"],
    border=CONFIG["add_white_border"],
    contrast_factor=CONFIG["contrast_factor"],
)
```

Save crop:

```text
/content/ppocr_line_perspective_crops/line_000_persp.png
```

### 7.2. Group crop

Sau khi group line boxes, crop group bằng axis-aligned bbox:

```python
group_crop = crop_axis_from_bbox(img_rgb, group_bbox_xyxy, pad=CONFIG["group_crop_padding"])
```

Save crop:

```text
/content/ppocr_stacked_group_crops/group_000.png
```

Group crop dùng cho Vintern group fallback.

---

## 8. Grouping stage

### 8.1. Điều kiện group 2 line

Hai line được group nếu:

```text
x_overlap_ratio >= group_min_x_overlap_ratio
vertical_gap <= avg_line_height * group_max_vertical_gap_ratio
```

Pseudo-code:

```python
def should_group_lines(a, b):
    x_ov = x_overlap_ratio(a["bbox_xyxy"], b["bbox_xyxy"])
    y_gap = vertical_gap(a["bbox_xyxy"], b["bbox_xyxy"])
    avg_h = (height(a) + height(b)) / 2

    return (
        x_ov >= CONFIG["group_min_x_overlap_ratio"]
        and y_gap <= avg_h * CONFIG["group_max_vertical_gap_ratio"]
    )
```

Sau đó dùng Union-Find để nối các line thành group.

### 8.2. Reading order

Group sort theo:

```python
(reading_order_y, reading_order_x)
```

Line trong group cũng sort top-to-bottom, left-to-right.

---

## 9. VietOCR recognition stage

### 9.1. Gọi VietOCR

```python
text, rec_conf, rec_conf_source = vietocr_predict_with_conf(crop)
```

Wrapper cần xử lý:

```python
out = vietocr_predictor.predict(crop, return_prob=True)
```

Các case:

```text
(text, prob)         → rec_conf_source = return_prob_valid
[text, prob]         → rec_conf_source = return_prob_list_valid
text only            → rec_conf_source = return_prob_no_conf
missing confidence   → rec_conf_source = return_prob_missing
exception/type error → fallback predict(crop)
```

Nếu không có confidence:

```python
rec_conf = None
```

Sau đó trong scoring:

```python
rec_conf_norm = CONFIG["rec_conf_fallback_when_missing"]  # default 0.50
```

### 9.2. Detect rec_conf flat

Sau khi nhận diện tất cả line:

```python
rec_conf_flat = (
    std(rec_confs) <= rec_conf_flat_std_threshold
    or unique_ratio <= rec_conf_flat_unique_ratio_threshold
)
```

Nếu `rec_conf_flat=True`, set cho từng line:

```python
line["rec_conf_flat_run"] = True
```

và recompute gating bằng rule `rec_missing`.

---

## 10. Wordlist + lexical features

### 10.1. Tokenization

```python
TOKEN_RE = r"[0-9A-Za-zÀ-ỹĐđ]+(?:[-'][0-9A-Za-zÀ-ỹĐđ]+)*"
```

Text OCR được tách thành token:

```python
tokens = extract_tokens(text)
eval_tokens = [t for t in tokens if len(t) >= 2 and not t.isdigit()]
```

### 10.2. `lex_ratio`

```python
lex_ratio = số eval_token nằm trong VN_WORDSET / tổng eval_token
```

Nếu không có wordlist đủ lớn:

```python
lex_ratio = neutral_lex_ratio_if_no_wordlist  # default 0.50
```

### 10.3. `diacritic_susp`

Nếu token không khớp wordlist nhưng bản không dấu của token tồn tại trong `VN_BASE_TO_VARIANTS`:

```text
token = "nguoi"
base = "nguoi"
base tồn tại, nhưng "nguoi" không phải variant hợp lệ/có dấu
→ nghi lỗi dấu
```

Tính:

```python
diacritic_susp = len(diacritic_suspicious_tokens) / len(eval_tokens)
```

### 10.4. `charset_penalty`

```python
charset_penalty = số ký tự không thuộc allowed charset / tổng ký tự
```

Dùng để phạt ký tự lạ/rác.

### 10.5. `repetition_penalty`

Phạt chuỗi lặp token/ký tự, ví dụ:

```text
"PP PP PP"
"aaaaaa"
```

---

## 11. Composite score

### 11.1. Formula hiện tại

```python
score = bias
score += 0.42 * rec_conf
score += 0.18 * det_score
score += 0.22 * lex_ratio
score += -0.12 * diacritic_susp
score += -0.06 * charset_penalty
score += -0.08 * repetition_penalty
score = clip(score, 0, 1)
```

Output:

```python
composite_score = score
quality_score = score * 100
```

### 11.2. Priority cho VLM

```python
priority = (1 - composite_score) + content_value_weight * content_value
```

Trong đó `content_value` tăng nhẹ theo số token/độ dài text.

Ý nghĩa:

```text
composite_score thấp → nghi ngờ → priority cao
text có nội dung dài hơn → priority tăng nhẹ
```

Vintern candidates được sort giảm dần theo `priority`.

---

## 12. Structural filter

Các text bị bỏ qua trước khi xét VLM:

```text
empty
timestamp
top-right logo
bottom counter
60 giây variants
```

### 12.1. Timestamp

```text
18:48:36
18.48.36
06:31
```

### 12.2. Top-right logo

Điều kiện:

```python
x1 > W * 0.72 and y2 < H * 0.22 and len(text) <= 10
```

### 12.3. 60 giây / bottom counter

Các biến thể:

```text
60
69
6o
giây
giay
giấy
gầy
(gầy
giy
```

Chỉ filter cứng nếu nằm ở lower-third:

```python
y1 >= H * 0.70
```

---

## 13. Gating decision

### 13.1. Standard rule khi `rec_conf` đáng tin

Auto accept nếu:

```python
rec_conf >= 0.90
det_score >= 0.70
lex_ratio >= 0.50
diacritic_susp <= 0.00
charset_penalty <= 0.05
```

Escalate nếu:

```python
weak_detection == True
or composite_score < 0.68
or rec_conf < 0.85
or lex_ratio < 0.45
or diacritic_susp > 0.30
```

### 13.2. Rule riêng khi `rec_conf` missing/flat

Nếu confidence không đáng tin, không auto-accept line ngắn.

Auto accept chỉ khi:

```python
token_count >= 5
det_score >= 0.78
lex_ratio >= 0.70
diacritic_susp <= 0.05
charset_penalty <= 0.02
weak_detection == False
```

Nếu không thỏa:

```text
vlm_candidate
```

Mục tiêu:

```text
ticker/caption dài, rõ, lex tốt → có thể auto_accept
biển đường/tên riêng ngắn → không auto_accept, gửi Vintern group
```

---

## 14. Vintern line fallback

### 14.1. Candidate selection

```python
vlm_candidates = [line for line in line_items if line["send_to_vintern"]]
candidates = sorted(vlm_candidates, key=lambda x: -x["priority"])[:vintern_max_candidates]
```

### 14.2. OCR prompt

```text
Hãy đọc chính xác toàn bộ chữ trong ảnh crop này.
Chỉ trả về nội dung OCR, không giải thích.
Giữ nguyên tiếng Việt có dấu nếu có.
```

### 14.3. Vintern preprocessing

Vintern dùng `dynamic_preprocess`:

```text
PIL image
→ chọn tile layout theo aspect ratio
→ resize mỗi tile về 448×448
→ normalize ImageNet mean/std
→ stack tensor [num_tiles, 3, 448, 448]
```

Điều này bắt buộc để tránh lỗi:

```text
NameError: dynamic_preprocess is not defined
```

### 14.4. Accept line Vintern

So sánh VietOCR và Vintern bằng:

```python
agreement_similarity = difflib.SequenceMatcher(...).ratio()
```

Nếu similarity cao:

```text
agree → chọn text có composite_score tốt hơn, need_review=False
```

Nếu disagreement:

```text
Vintern chỉ override nếu:
vintern_score >= vintern_min_composite_accept
và vintern_score >= old_score + margin

Nếu override do disagreement:
need_review=True
```

Nếu Vintern không đủ tốt:

```text
giữ VietOCR, keep_for_index=False, need_review=True
```

---

## 15. Vintern group fallback

### 15.1. Khi nào gọi group Vintern?

Group được gửi Vintern nếu:

```python
use_vintern_group_fallback == True
group_crop_path tồn tại
num_lines >= vintern_group_min_lines
và group có ít nhất 1 line need_review hoặc send_to_vintern
```

### 15.2. Group priority

```python
group_vlm_priority = mean(line.priority) + bonus
```

Bonus:

```text
+0.25 nếu group nhiều dòng
+0.20 nếu có line VLM candidate
+0.15 nếu có need_review
```

Chọn top:

```python
vintern_group_max_candidates = 4
```

### 15.3. Group OCR prompt

```text
Hãy đọc chính xác toàn bộ chữ trong ảnh crop này theo đúng từng dòng.
Nếu ảnh có nhiều dòng chữ, hãy xuống dòng giữa các dòng.
Chỉ trả về nội dung OCR, không giải thích.
```

### 15.4. Accept group Vintern

Accept nếu:

```python
group_vintern_composite_score >= 0.55
```

Hoặc rule mềm:

```python
len(group_vintern_text) >= max(6, len(group_text_review) * 0.6)
lex_ratio >= 0.45
charset_penalty <= 0.05
```

Nếu accept:

```python
group_text_clean = group_vintern_text
group_text = group_vintern_text
group_final_source = "vintern_group_fallback"
group_keep_for_index = True
need_review = similarity_with_old_review < 0.82
```

Nghĩa là:

```text
Vintern group có thể được index,
nhưng nếu khác nhiều với VietOCR line-level thì vẫn need_review=True.
```

---

## 16. Text outputs

### 16.1. Clean output

Chỉ dùng để index/search:

```python
final_clean_text = "\n\n".join(clean_group_texts)
final_search_text = final_clean_text
```

Nguồn của clean text:

```text
line auto_accept
line Vintern accepted
group Vintern accepted
```

### 16.2. Review output

Không dùng để index chính:

```python
final_review_text = "\n\n".join(review_group_texts)
```

Nguồn:

```text
text nghi ngờ
text disagreement
text chưa đủ score
VietOCR/Vintern candidates chưa được accept
```

### 16.3. Rule quan trọng

Không được làm:

```python
group_text = clean_text if clean_text else review_text
```

Phải làm:

```python
group_text = group_text_clean
group_text_review = review_text
```

---

## 17. Export format

Output folder:

```text
/content/ppocr_group_vietocr_vintern_wordlist_gating_v2_output
```

Files:

```text
<frame>_ocr_lines_wordlist_gating_v2.csv
<frame>_ocr_groups_wordlist_gating_v2.csv
<frame>_ocr_es_doc_wordlist_gating_v2.json
<frame>_ocr_clean_text_wordlist_gating_v2.txt
<frame>_ocr_review_text_wordlist_gating_v2.txt
<frame>_ocr_vis_wordlist_gating_v2.png
```

### 17.1. CSV line-level

Nên có các cột:

```text
line_id
group_id
vietocr_text
rec_conf
rec_conf_source
rec_conf_flat_run
det_score
composite_score
quality_score
lex_ratio
diacritic_susp
diacritic_suspicious_tokens
oov_tokens
charset_penalty
repetition_penalty
weak_detection
priority
filter_status
send_to_vintern
vintern_text
vintern_composite_score
agreement_similarity
final_text
final_source
keep_for_index
need_review
persp_crop_path
group_crop_path
bbox_xyxy
```

### 17.2. CSV group-level

Nên có các cột:

```text
reading_order
group_id
num_lines
group_text
group_text_clean
group_text_review
group_vintern_text
group_vintern_composite_score
group_vintern_agreement_similarity
group_final_source
group_keep_for_index
need_review
group_crop_path
bbox_xyxy
mean_det_score
mean_composite_score
num_keep_lines
num_vlm_candidates
```

### 17.3. ES document

Schema chính:

```json
{
  "frame_id": "...",
  "image_path": "...",
  "ocr_pipeline": {
    "detector": "PP-OCRv6_medium_det",
    "line_recognizer": "VietOCR/vgg_transformer",
    "fallback_vlm_line": "5CD-AI/Vintern-1B-v3_5",
    "fallback_vlm_group": "5CD-AI/Vintern-1B-v3_5",
    "wordlist_enabled": true,
    "wordlist_size": 123456,
    "group_text_rule": "group_text_clean only; group_text_review is not indexed"
  },
  "timing": {
    "det_time_sec": 0.0,
    "vietocr_total_time_sec": 0.0,
    "vintern_line_total_time_sec": 0.0,
    "vintern_group_total_time_sec": 0.0
  },
  "ocr_text_clean": "...",
  "ocr_text_review": "...",
  "ocr_group_texts_clean": [],
  "ocr_group_texts_review": [],
  "ocr_lines": [],
  "ocr_groups": []
}
```

---

## 18. Agent implementation checklist

### 18.1. Environment

Agent cần đảm bảo:

```bash
pip install paddleocr paddlepaddle
pip install vietocr
pip install transformers==4.45.2 accelerate==0.34.2 tokenizers==0.20.3
pip install timm einops torchvision pillow opencv-python-headless pandas matplotlib tqdm
```

Nếu dùng Colab GPU:

```python
torch.cuda.is_available() == True
```

Paddle có thể chạy CPU nếu GPU Paddle lỗi:

```python
PADDLE_DEVICE = "gpu:0" if paddle.is_compiled_with_cuda() else "cpu"
```

### 18.2. Wordlist

Agent nên kiểm tra wordlist trước khi chạy:

```python
assert Path("/content/drive/MyDrive/vietnamese/general_dict.txt").exists()
assert Path("/content/drive/MyDrive/vietnamese/vn_dictionary.txt").exists()
```

Nếu không có:

```text
WORDLIST_ENABLED=False
lex_ratio=0.50 neutral
```

Điều này vẫn chạy được nhưng gating kém chính xác hơn.

### 18.3. Vintern

Agent phải đảm bảo sau khi load Vintern:

```python
dynamic_preprocess is defined
vintern_model is not None
vintern_tokenizer is not None
```

Nếu Vintern inference time chỉ `0.003s/crop`, gần như chắc chắn Vintern không chạy thật mà đang lỗi nhanh.

### 18.4. Debug quan trọng

Sau Cell 12 cần kiểm tra:

```text
rec_conf_source distribution
rec_conf stats
rec_conf_flat
num_auto_accept
num_escalate_candidates
escalation_rate_non_structural
```

Nếu:

```text
rec_conf_source = return_prob_missing toàn bộ
```

thì pipeline đang dùng rec-missing rule.

### 18.5. Chỉ số cần log khi scale nhiều frame

Per frame:

```text
num_lines
num_structural_filtered
num_auto_accept
num_line_vlm_candidates
num_line_vintern_called
num_group_vintern_called
num_keep_lines
num_review_groups
det_time
vietocr_time
vintern_line_time
vintern_group_time
```

Aggregate:

```text
auto_accept_rate
line_escalation_rate
group_escalation_rate
review_rate
avg_vietocr_time_per_crop
avg_vintern_time_per_line
avg_vintern_time_per_group
```

---

## 19. Pseudocode end-to-end

```python
def run_ocr_pipeline(image_path):
    img_rgb = load_image(image_path)

    boxes, det_scores = ppocr_detect(img_rgb)

    line_items = []
    for box, det_score in zip(boxes, det_scores):
        if not valid_box(box, det_score):
            continue

        line_crop = crop_line_perspective(img_rgb, box)
        line_item = make_line_item(box, det_score, line_crop)
        line_items.append(line_item)

    groups = group_line_items(line_items)

    for group in groups:
        group_crop = crop_group_context(img_rgb, group)
        group["group_crop_path"] = save(group_crop)

    for line in line_items:
        text, rec_conf, source = vietocr_recognize(line["persp_crop_path"])
        line["vietocr_text"] = text
        line["rec_conf"] = rec_conf
        line["rec_conf_source"] = source
        score_and_gate_line(line)

    if rec_conf_is_flat_or_missing(line_items):
        for line in line_items:
            line["rec_conf_flat_run"] = True
            score_and_gate_line(line)

    line_candidates = rank_line_vlm_candidates(line_items)

    for line in line_candidates:
        vtext = vintern_ocr_line(line["persp_crop_path"])
        decide_line_final_text(line, vtext)

    build_group_text_clean_review(groups, line_items)

    group_candidates = rank_group_vlm_candidates(groups, line_items)

    for group in group_candidates:
        gv_text = vintern_ocr_group(group["group_crop_path"])
        decide_group_final_text(group, gv_text)

    final_clean_text = collect_group_clean_text(groups)
    final_review_text = collect_group_review_text(groups)

    export_outputs(line_items, groups, final_clean_text, final_review_text)
```

---

## 20. Các lỗi thường gặp và cách xử lý

### 20.1. `NameError: dynamic_preprocess is not defined`

Nguyên nhân: chưa chạy cell định nghĩa Vintern preprocess.

Fix:

```text
Chạy lại CELL 6.1 trước CELL 12/13.1.
```

### 20.2. `rec_conf_source = return_prob_missing`

Nguyên nhân: VietOCR version/config không trả confidence.

Hướng xử lý:

```text
Pipeline vẫn chạy bằng rec-missing rule.
Có thể thử vietocr_beamsearch=False để xem return_prob có hoạt động không.
```

### 20.3. Escalation rate = 1.0

Nếu confidence missing toàn bộ, các dòng ngắn/tên riêng sẽ bị escalate là đúng. Nhưng ticker dài nên có thể auto-accept nếu wordlist tốt.

Tuning:

```python
CONFIG["rec_missing_auto_accept_min_tokens"] = 4
CONFIG["rec_missing_auto_accept_lex_ratio"] = 0.60
```

### 20.4. Group Vintern không accept dù đọc đúng

Tuning:

```python
CONFIG["group_vintern_min_composite_accept"] = 0.50
```

hoặc sửa `group_vintern_should_accept`.

### 20.5. Text review lọt vào index

Không được fallback:

```python
group_text = clean_text if clean_text else review_text
```

Phải giữ:

```python
group_text = group_text_clean
```

---

## 21. Roadmap sau khi agent implement xong

### P0 — chạy ổn single frame

- Detector chạy được.
- Line crop/group crop lưu đúng.
- VietOCR chạy được.
- Wordlist load được.
- Gating tạo `clean_text` và `review_text` riêng.
- Vintern line/group chạy thật.

### P1 — benchmark vài nghìn frame

Log:

```text
escalation_rate
review_rate
false_accept samples
false_reject samples
runtime per stage
```

### P2 — tối ưu rule tay

Dựa trên log thật:

```text
chỉnh threshold rec_missing
chỉnh group_vintern acceptance
chỉnh structural filter
```

### P3 — train composite score

Chỉ làm sau khi có khoảng 200–300 dòng gán nhãn thật từ nhiều video:

```text
label: correct / incorrect / review / structural
features: rec_conf, det_score, lex_ratio, diacritic_susp, charset_penalty, text_len, token_count
model: logistic regression hoặc small tree
```

Không nên train trước khi có log thật, vì dễ overfit một clip.

---

## 22. Kết luận triển khai

Workflow hiện tại nên được hiểu là **OCR confidence funnel**, không chỉ là OCR model chaining.

Vai trò từng thành phần:

```text
PP-OCRv6 DET:
  phát hiện line boxes

VietOCR:
  recognizer chính, nhanh hơn VLM

Wordlist + composite score:
  giảm escalation rate giả
  phát hiện nghi lỗi dấu
  quyết định auto_accept hay VLM

Vintern line fallback:
  xử lý line crop khó

Vintern group fallback:
  xử lý group nhiều dòng/cần context

group_text_clean:
  output an toàn để index

group_text_review:
  output debug/manual review, không index chính
```

Nguyên tắc quan trọng nhất:

```text
Không index text nếu chưa được accept.
Nếu không chắc → review hoặc Vintern group.
```
