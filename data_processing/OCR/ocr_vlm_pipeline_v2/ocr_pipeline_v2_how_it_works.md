# OCR VLM Pipeline v2 - Cách hoạt động và xử lý dữ liệu

Tài liệu này giải thích pipeline OCR VLM v2 theo hướng dễ hiểu: input đi vào đâu, từng stage xử lý gì, vì sao cần VietOCR/Vintern/wordlist, output cuối được tạo như thế nào, và khi chạy nhiều frame thì pipeline tối ưu ra sao.

File workflow/report chi tiết hơn nằm ở:

```text
current_ocr_pipeline_workflow_agent_plan.md
```

---

## 1. Ý tưởng tổng quát

Pipeline hiện tại không gửi toàn bộ ảnh vào VLM để đọc tất cả chữ. Thay vào đó, pipeline dùng nhiều tầng xử lý:

```text
1. PP-OCRv6 detect vị trí chữ
2. Crop từng dòng chữ
3. VietOCR đọc nhanh từng crop
4. Wordlist + scoring đánh giá độ tin cậy
5. Chỉ gửi crop nghi ngờ sang Vintern VLM
6. Gom clean text để index/search
7. Đẩy text nghi ngờ sang review
```

Lý do thiết kế như vậy:

- PP-OCRv6 detect nhanh và tốt cho text box.
- VietOCR đọc line crop nhanh hơn VLM rất nhiều.
- Vintern/VLM đắt hơn, nên chỉ dùng cho những dòng/cụm khó.
- Wordlist giúp phát hiện lỗi dấu, ký tự lạ, text nhiễu.
- `clean_text` và `review_text` được tách riêng để tránh index nhầm text chưa chắc đúng.

Pipeline này nên được hiểu là một **confidence funnel**:

```text
nhiều text box
-> đọc nhanh bằng VietOCR
-> lọc noise
-> đánh giá độ tin cậy
-> chỉ giữ text chắc
-> text khó mới dùng VLM
```

---

## 2. Input và output

### 2.1. Input

Input cơ bản là một frame ảnh:

```text
/path/to/frame.jpg
```

Ví dụ:

```text
/tmp2/maitanha/vgu/ttn/data/AIC2025/keyframes/Keyframes_L27/L27_V010/050.jpg
```

Khi chạy batch, input là folder chứa nhiều frame:

```text
/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test/L22_V012/
```

### 2.2. Output single-frame

Khi chạy `run_single_frame.py`, pipeline thường xuất đủ artifact để debug:

```text
outputs_test/<frame_run>/
├── <frame>_ocr_lines_wordlist_gating_v2.csv
├── <frame>_ocr_groups_wordlist_gating_v2.csv
├── <frame>_ocr_clean_text_wordlist_gating_v2.txt
├── <frame>_ocr_review_text_wordlist_gating_v2.txt
├── <frame>_ocr_es_doc_wordlist_gating_v2.json
├── <frame>_ocr_vis_wordlist_gating_v2.png
├── ppocr_line_perspective_crops/
└── ppocr_stacked_group_crops/
```

### 2.3. Output batch video

Khi chạy `run_frame_folder.py` với `--batch_artifacts minimal`, output chính là:

```text
outputs_video/<video_id>/
├── <video_id>_ocr_es_docs.jsonl
├── <video_id>_ocr_frame_summary.csv
└── <video_id>_ocr_timing_summary.json
```

Đây là output phù hợp để scale lên nhiều frame/video.

---

## 3. Các file chính

```text
run_single_frame.py
```

Chạy toàn bộ pipeline trên 1 frame. Dùng để debug chất lượng.

```text
run_frame_folder.py
```

Chạy pipeline trên cả folder frame. Dùng cho một video hoặc nhiều frame đã extract.

```text
search_ocr_results.py
```

Search thử trên JSONL output mà chưa cần Elasticsearch thật.

```text
ocr_pipeline/config.py
```

Chứa toàn bộ threshold/config mặc định.

```text
ocr_pipeline/detector.py
```

Load PP-OCRv6, parse output detector, filter box.

```text
ocr_pipeline/cropping.py
```

Crop line bằng perspective và crop group bằng bbox.

```text
ocr_pipeline/grouping.py
```

Gộp các text line gần nhau thành group.

```text
ocr_pipeline/recognizers/vietocr_recognizer.py
```

Wrapper cho VietOCR.

```text
ocr_pipeline/recognizers/vintern_recognizer.py
```

Wrapper cho Vintern VLM line/group fallback.

```text
ocr_pipeline/scoring/
```

Tính wordlist features, composite score, gating decision.

```text
ocr_pipeline/structural_filter.py
```

Lọc timestamp, logo, counter, text noise/hallucination.

```text
ocr_pipeline/outputs.py
```

Gom clean/review text, export CSV/JSON/JSONL/visualization.

---

## 4. Stage 1 - Load image và wordlist

Pipeline đọc frame bằng OpenCV:

```text
image_path
-> cv2.imread
-> BGR image
-> RGB image
```

Sau đó load wordlist:

```text
word_list/vn_dictionary.txt
word_list/general_dict.txt
```

Wordlist dùng để biết token nào có vẻ hợp lệ trong tiếng Việt hoặc text phổ biến.

Ví dụ:

```text
"nhiễm" nằm trong wordlist -> tốt
"NHIEM" có thể là không dấu nhưng vẫn có base hợp lệ
"AIIIAAIAIIII" -> không giống token thật, dễ bị xem là noise
```

Nếu wordlist không load được hoặc quá nhỏ, pipeline vẫn chạy nhưng lexical score sẽ kém tin cậy hơn.

---

## 5. Stage 2 - PP-OCRv6 detect text box

Model detect:

```text
PP-OCRv6_medium_det
```

Detector chỉ làm nhiệm vụ tìm vị trí chữ, chưa đọc nội dung.

Output mong muốn:

```text
box polygon 4 điểm
det_score
```

Ví dụ line item sau detect:

```json
{
  "line_id": 0,
  "det_idx": 3,
  "box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "bbox_xyxy": [x1, y1, x2, y2],
  "det_score": 0.93
}
```

### 5.1. Detector fallback API

Do version PaddleOCR/PaddleX khác nhau, code thử nhiều cách tạo detector:

```text
paddleocr.TextDetection
-> paddlex.TextDetection
-> PaddleOCR end-to-end detector
-> paddlex.create_model
```

Mục tiêu là cùng một pipeline chạy được trên nhiều môi trường hơn.

### 5.2. Filter box

Sau detect, box bị lọc nếu:

```text
score quá thấp
width quá nhỏ
height quá nhỏ
aspect không hợp lý
```

Điểm quan trọng: aspect hiện tại tính bằng:

```text
width / height
```

Không dùng:

```text
min(width, height) / max(width, height)
```

Lý do: subtitle/ticker dài phía dưới thường rất rộng và thấp. Nếu dùng `min/max`, các box này sẽ bị drop nhầm.

---

## 6. Stage 3 - Crop line

Mỗi detected box được crop thành một ảnh nhỏ.

Quy trình:

```text
polygon box
-> perspective transform
-> padding
-> upscale nếu crop quá thấp
-> tăng contrast
-> thêm white border
-> save crop
```

Output:

```text
ppocr_line_perspective_crops/line_000_persp.png
```

Line crop này là input cho:

- VietOCR
- Vintern line fallback nếu dòng đó khó

Lý do phải crop:

- VietOCR đọc tốt hơn khi chỉ nhìn một dòng chữ.
- VLM cũng chính xác hơn khi crop tập trung vào vùng chữ, thay vì nhìn nguyên frame nhiều nhiễu.

---

## 7. Stage 4 - Group line boxes

Các line gần nhau được gộp thành group.

Một group có thể là:

- bảng hiệu nhiều dòng
- subtitle/lower-third
- block text trong biển báo
- cụm text trên màn hình

Hai line được group nếu:

```text
chồng nhau đủ theo trục x
và khoảng cách dọc không quá xa
```

Config chính:

```text
group_min_x_overlap_ratio = 0.12
group_max_vertical_gap_ratio = 1.80
```

Sau khi group, pipeline crop một ảnh group:

```text
ppocr_stacked_group_crops/group_000.png
```

Group crop dùng cho Vintern group fallback. Đây là bước quan trọng vì nhiều khi một dòng riêng lẻ thiếu context, nhưng cả group thì VLM đọc tốt hơn.

---

## 8. Stage 5 - VietOCR recognition

VietOCR đọc từng line crop.

Input:

```text
line crop image
```

Output:

```text
vietocr_text
rec_conf nếu lấy được
rec_conf_source
```

Ví dụ:

```json
{
  "vietocr_text": "KHOA NHIỄM - THẦN KINH",
  "rec_conf": 0.91,
  "rec_conf_source": "return_prob_valid"
}
```

Nếu VietOCR không trả confidence:

```text
rec_conf = None
rec_conf_source = return_prob_missing / return_prob_no_conf
```

Pipeline khi đó dùng fallback:

```text
rec_conf_fallback_when_missing = 0.50
```

và chuyển gating sang rule thận trọng hơn.

### 8.1. Vì sao hiện tại khuyến nghị `--no_vietocr_batch`

Pipeline có experimental batch path cho VietOCR, nhưng từng gặp output nhiễu dài:

```text
AIIIAAIAIIII...
1B0CBC2BBC...
NN1(N1N...
```

Do đó, khi chạy chính nên dùng:

```bash
--no_vietocr_batch
```

để ưu tiên official VietOCR predictor path ổn định.

---

## 9. Stage 6 - Scoring và gating

Sau khi VietOCR đọc text, pipeline không index ngay. Nó đánh giá chất lượng trước.

Mỗi line được tính:

```text
lex_ratio
diacritic_susp
charset_penalty
repetition_penalty
det_score
rec_conf
composite_score
priority
```

### 9.1. lex_ratio

Tỷ lệ token hợp lệ theo wordlist.

Ví dụ:

```text
"KHOA NHIỄM THẦN KINH" -> lex_ratio cao
"AIIIAAIAIIII" -> lex_ratio thấp
```

### 9.2. diacritic_susp

Đo nghi ngờ lỗi dấu tiếng Việt.

Ví dụ:

```text
"nguoi" có base tiếng Việt nhưng thiếu dấu -> có thể tăng suspicion
```

### 9.3. charset_penalty

Phạt ký tự lạ:

```text
�, [], (), symbol quá nhiều, ký tự không thuộc charset mong muốn
```

### 9.4. repetition_penalty

Phạt text lặp bất thường:

```text
AAAAIIIIAAAA
BBBBCCCCBBBB
```

### 9.5. Gating statuses

Pipeline gán một trong các trạng thái:

| Status | Ý nghĩa |
|---|---|
| `auto_accept` | Đủ tin cậy, đưa vào clean text. |
| `vlm_candidate` | Nghi ngờ, có thể gửi Vintern. |
| `structural_filter` | Text thuộc timestamp/logo/counter/noise, bỏ. |
| `review_only` | Giữ để review, không chắc hoàn toàn. |
| `text_noise_filter` | Bị loại sau khi phát hiện noise ở final text. |

---

## 10. Stage 7 - Structural filter và noise filter

Structural filter loại các text không nên index:

```text
empty
timestamp
logo top-right
bottom counter
60 giây variants
noise text
```

Ví dụ timestamp:

```text
18:30:11
06:59:49
```

Ví dụ noise:

```text
AIIIAAIAIIIIIIIAAAAIAIIIAAIAIA...
1B0CBC2BBCBGCBGG0BCC0...
```

Filter này rất quan trọng để tránh JSONL/ES bị nhiễu và search trả kết quả rác.

---

## 11. Stage 8 - Vintern VLM fallback

Vintern không chạy cho tất cả crop. Nó chỉ chạy khi line/group đáng nghi.

Vintern nhận:

```text
image crop + prompt
```

Không nhận text box thô. Không tự detect text.

Có 2 loại input:

| Loại | Input image | Prompt | Mục đích |
|---|---|---|---|
| Line fallback | line crop | đọc chính xác chữ trong crop | Sửa một dòng khó |
| Group fallback | group crop | đọc chính xác nhiều dòng, giữ xuống dòng | Sửa cụm text cần context |

### 11.1. Prompt line

```text
Hãy đọc chính xác toàn bộ chữ trong ảnh crop này.
Chỉ trả về nội dung OCR, không giải thích.
Giữ nguyên tiếng Việt có dấu nếu có.
```

### 11.2. Prompt group

```text
Hãy đọc chính xác toàn bộ chữ trong ảnh crop này theo đúng từng dòng.
Nếu ảnh có nhiều dòng chữ, hãy xuống dòng giữa các dòng.
Chỉ trả về nội dung OCR, không giải thích.
```

### 11.3. Preprocess ảnh cho Vintern

Vintern crop được xử lý như sau:

```text
PIL image
-> dynamic_preprocess
-> resize/split thành tile 448x448
-> normalize theo ImageNet mean/std
-> tensor float16
-> model.chat(tokenizer, pixel_values, "<image>\n<prompt>", generation_config)
```

### 11.4. Quyết định accept Vintern

Sau khi Vintern trả text:

```text
so sánh VietOCR text với Vintern text
tính composite score cho Vintern text
kiểm tra noise
quyết định accept/override/review
```

Nếu VietOCR và Vintern giống nhau:

```text
agreement cao -> chọn text tốt hơn -> keep_for_index=True
```

Nếu khác nhau:

```text
Vintern chỉ override nếu score đủ tốt và tốt hơn VietOCR đủ margin
```

Nếu Vintern sinh text noise:

```text
reject Vintern -> không index text đó
```

---

## 12. Stage 9 - Gom clean text và review text

Sau khi line và group đã có final decision, pipeline gom text theo group.

Mỗi group có:

```text
group_text_clean
group_text_review
group_text
```

Rule bắt buộc:

```text
group_text = group_text_clean
```

Không được fallback:

```text
group_text = clean nếu có, ngược lại review
```

Lý do: review text chưa chắc đúng, không nên vào search/index chính.

Cuối cùng:

```text
final_clean_text = nối các group_text_clean
final_review_text = nối các group_text_review
```

---

## 13. Stage 10 - Export ES-friendly document

Pipeline tạo document dạng thân thiện với Elasticsearch.

Các field quan trọng:

```text
document_id
video_id
frame_id
frame_number
image_path
ocr_text_clean
ocr_text_review
ocr_text_search
ocr_text_unaccent
ocr_terms
ocr_lines
ocr_groups
quality
timing
```

Ý nghĩa:

| Field | Vai trò |
|---|---|
| `ocr_text_clean` | Text đã được accept. |
| `ocr_text_review` | Text nghi ngờ, dùng debug/audit. |
| `ocr_text_search` | Text clean dùng search chính. |
| `ocr_text_unaccent` | Bản không dấu cho fuzzy/lazy search tiếng Việt. |
| `ocr_terms` | Token unique phục vụ search/filter/autocomplete đơn giản. |
| `ocr_lines` | Evidence line-level. |
| `ocr_groups` | Evidence group-level. |
| `quality` | Thống kê chất lượng frame. |
| `timing` | Thời gian từng stage. |

Mặc định:

```text
review text không được đưa vào search chính
```

---

## 14. Batch processing hoạt động như thế nào

`run_frame_folder.py` xử lý folder frame theo thứ tự tự nhiên:

```text
001.jpg
002.jpg
003.jpg
...
```

Với `workers=1`:

```text
load model 1 lần
for từng frame:
  run_ocr_pipeline
  append JSONL
  update summary
```

Với `workers>1`:

```text
spawn nhiều process
mỗi process load model riêng
parent process nhận kết quả frame hoàn thành
append JSONL
ghi summary cuối run
```

Batch output không cần giữ artifact nặng nếu dùng:

```bash
--batch_artifacts minimal --cleanup_crops
```

### 14.1. Vì sao cần progress/ETA

Pipeline có thể chạy hàng giờ/ngày khi nhiều frame. Vì vậy batch runner hiển thị:

```text
frame hiện tại / tổng frame
avg sec/frame
ETA
last frame id
```

Ví dụ:

```text
OCR L22_V012_full_w4: 100%| 283/283 [05:29<00:00, 1.16s/frame, avg=1.16s ETA=0s last=227]
```

---

## 15. Search local hoạt động như thế nào

`search_ocr_results.py` đọc JSONL và tính điểm match query.

Nó dùng:

```text
ocr_text_search
ocr_text_clean
ocr_group_texts_clean
```

Nếu bật `--include_review`, nó đọc thêm:

```text
ocr_text_review
ocr_group_texts_review
```

Search hỗ trợ:

- bỏ dấu tiếng Việt
- fuzzy token match
- exact phrase match
- show line evidence
- lọc frame range

Ví dụ:

```bash
python search_ocr_results.py \
  --jsonl outputs_video/L22_V012_full_w4 \
  --query "khoa nhiễm thần kinh" \
  --show_lines
```

---

## 16. Cách xử lý các tình huống thường gặp

### 16.1. Không detect được subtitle phía dưới

Kiểm tra:

- `det_limit_side_len`
- `det_thresh`
- `det_box_thresh`
- `det_unclip_ratio`
- filter aspect

Quan trọng nhất: đảm bảo aspect đang là:

```text
width / height
```

Nếu subtitle đã detect nhưng bị drop, xem log:

```text
After filtering: ... reasons={...}
```

### 16.2. VietOCR sinh text rác dài

Nếu đang bật batch decoder:

```bash
--vietocr_batch
```

hãy chuyển về:

```bash
--no_vietocr_batch
```

Nếu vẫn có rác, kiểm tra `noise_filter_*` trong config.

### 16.3. VLM chạy quá lâu

Kiểm tra:

```text
vintern_max_candidates
vintern_group_max_candidates
gating_stats
num_vlm_candidates
```

Giảm số candidate nếu cần:

```bash
--vintern_max_candidates 4
--vintern_group_max_candidates 2
```

### 16.4. Search không ra kết quả mong muốn

Kiểm tra:

```text
ocr_text_clean có text không
text bị đưa vào review không
text bị noise filter không
query có dấu/không dấu
```

Thử:

```bash
python search_ocr_results.py \
  --jsonl outputs_video/<video_id> \
  --query "<query>" \
  --include_review \
  --show_lines
```

Nếu chỉ ra khi `--include_review`, nghĩa là text đó chưa được accept vào clean index.

### 16.5. Disk tăng quá nhanh

Khi chạy nhiều frame, luôn dùng:

```bash
--batch_artifacts minimal --cleanup_crops
```

Không bật visualization/crops cho full dataset.

---

## 17. Tuning nhanh

### 17.1. Muốn detect nhiều text nhỏ hơn

Thử tăng:

```python
det_limit_side_len = 1216
```

hoặc giảm nhẹ:

```python
det_box_thresh = 0.40
```

Đổi này có thể tăng noise và thời gian.

### 17.2. Muốn giảm số crop gửi Vintern

Tăng độ khó để escalate:

```python
escalate_composite_below
escalate_rec_conf_below
rec_missing_auto_accept_min_tokens
rec_missing_auto_accept_min_lex_ratio
```

Nhưng nếu quá chặt, có thể index nhầm text VietOCR sai.

### 17.3. Muốn Vintern đọc nhiều hơn để tăng recall

Tăng:

```python
vintern_max_candidates
vintern_group_max_candidates
```

Đổi này làm chậm pipeline.

### 17.4. Muốn batch chạy nhanh hơn

Thử:

```bash
--workers 2
--workers 3
--workers 4
```

Chọn mức có:

```text
avg sec/frame thấp
GPU util cao
không OOM
không làm disk I/O nghẽn
```

Nếu GPU util đã 100%, tăng worker thường không giúp nhiều.

---

## 18. Lệnh chạy mẫu

### 18.1. Chạy 1 frame đầy đủ Vintern

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/OCR/ocr_vlm_pipeline_v2

CUDA_VISIBLE_DEVICES=0 OCR_V2_PADDLE_DEVICE=gpu:0 OCR_V2_PADDLE_ENGINE=paddle_dynamic OCR_V2_VINTERN_ATTN=flash_attention_2 \
python run_single_frame.py \
  --image /tmp2/maitanha/vgu/ttn/data/AIC2025/keyframes/Keyframes_L27/L27_V010/050.jpg \
  --output_dir ./outputs_test/L27_V010_050 \
  --verbose
```

### 18.2. Chạy 1 frame không Vintern

```bash
CUDA_VISIBLE_DEVICES=0 OCR_V2_PADDLE_DEVICE=gpu:0 OCR_V2_PADDLE_ENGINE=paddle_dynamic \
python run_single_frame.py \
  --image /tmp2/maitanha/vgu/ttn/data/AIC2025/keyframes/Keyframes_L27/L27_V010/050.jpg \
  --output_dir ./outputs_test/L27_V010_050_no_vintern \
  --no_vintern \
  --verbose
```

### 18.3. Smoke test 40 frame

```bash
CUDA_VISIBLE_DEVICES=0 OCR_V2_PADDLE_DEVICE=gpu:0 OCR_V2_PADDLE_ENGINE=paddle_dynamic OCR_V2_VINTERN_ATTN=flash_attention_2 \
python run_frame_folder.py \
  --frames_dir ../../../keyframe_test/L22_V012 \
  --output_dir ./outputs_video \
  --video_id L22_V012_smoke_w4 \
  --limit 40 \
  --workers 4 \
  --no_vietocr_batch \
  --batch_artifacts minimal \
  --cleanup_crops \
  --quiet
```

### 18.4. Full video

```bash
CUDA_VISIBLE_DEVICES=0 OCR_V2_PADDLE_DEVICE=gpu:0 OCR_V2_PADDLE_ENGINE=paddle_dynamic OCR_V2_VINTERN_ATTN=flash_attention_2 \
python run_frame_folder.py \
  --frames_dir ../../../keyframe_test/L22_V012 \
  --output_dir ./outputs_video \
  --video_id L22_V012_full_w4 \
  --pattern "*.jpg" \
  --workers 4 \
  --no_vietocr_batch \
  --batch_artifacts minimal \
  --cleanup_crops \
  --quiet
```

### 18.5. Search thử

```bash
python search_ocr_results.py \
  --jsonl outputs_video/L22_V012_full_w4 \
  --query "khoa nhiễm thần kinh" \
  --top_k 10 \
  --show_lines
```

---

## 19. Tóm tắt ngắn

Pipeline xử lý theo nguyên tắc:

```text
Detect trước, đọc sau.
VietOCR đọc nhanh, Vintern chỉ sửa case khó.
Wordlist và scoring quyết định text có đủ tin cậy không.
Noise/timestamp/logo/counter bị lọc.
Clean text dùng để search/index.
Review text dùng để debug/audit.
Batch mode phải hạn chế artifact để không quá tải disk.
```

Luồng cuối cùng:

```text
Frame
-> PP-OCRv6 boxes
-> line crops
-> groups
-> VietOCR text
-> score/gate/filter
-> Vintern fallback nếu cần
-> clean/review text
-> ES JSONL
-> search local hoặc index Elasticsearch
```
