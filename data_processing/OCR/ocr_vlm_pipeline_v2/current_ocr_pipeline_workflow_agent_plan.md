# Report: OCR VLM Pipeline v2 - Workflow hiện tại

Tài liệu này mô tả workflow thật đang được implement trong `ocr_vlm_pipeline_v2`, không còn là bản kế hoạch từ notebook. Pipeline hiện tại được thiết kế để:

- Debug chất lượng trên 1 frame bằng `run_single_frame.py`.
- Chạy batch trên folder frame/video bằng `run_frame_folder.py`.
- Dùng PP-OCRv6 để detect text box.
- Dùng VietOCR làm recognizer chính.
- Dùng wordlist, composite score và structural filter để quyết định text nào đủ tin cậy.
- Chỉ gửi các crop nghi ngờ sang Vintern VLM.
- Xuất JSONL dạng ES-friendly để search/index.
- Có script search local trước khi đưa dữ liệu vào Elasticsearch thật.

Notebook gốc tham khảo:

```text
data_processing/OCR/ppocrv6_det_group_line_vietocr_vintern_wordlist_gating_v2.ipynb
```

Code hiện tại:

```text
data_processing/OCR/ocr_vlm_pipeline_v2/
├── run_single_frame.py
├── run_frame_folder.py
├── search_ocr_results.py
└── ocr_pipeline/
    ├── config.py
    ├── detector.py
    ├── cropping.py
    ├── grouping.py
    ├── structural_filter.py
    ├── outputs.py
    ├── recognizers/
    │   ├── vietocr_recognizer.py
    │   └── vintern_recognizer.py
    └── scoring/
        ├── features.py
        ├── gating.py
        └── wordlist.py
```

---

## 1. Mục tiêu workflow

Pipeline không phải là chuỗi OCR đơn giản. Nó là một **OCR confidence funnel**:

```text
Frame gốc
-> PP-OCRv6 detect text boxes
-> crop từng line
-> group các line gần nhau
-> VietOCR đọc tất cả line crop
-> wordlist + feature scoring
-> structural/noise filter
-> gating:
   - auto_accept
   - vlm_candidate
   - structural_filter
   - review_only
-> Vintern đọc line/group crop nếu cần
-> tạo clean text và review text
-> export CSV/TXT/PNG/JSON hoặc JSONL batch
-> search local hoặc index ES
```

Nguyên tắc quan trọng:

```text
Chỉ `clean text` được dùng cho search/index chính.
`review text` chỉ dùng để debug, audit hoặc search phụ khi cần.
Không được fallback group_text sang review_text.
```

---

## 2. Entry points

### 2.1. Debug 1 frame

File:

```text
run_single_frame.py
```

Dùng khi cần xem kỹ:

- detector có bắt đúng text không
- crop có cắt mất chữ/dấu không
- VietOCR đọc gì
- line/group nào bị gửi Vintern
- clean/review text ra sao
- visualization PNG có hợp lý không

Lệnh mẫu trên server:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/OCR/ocr_vlm_pipeline_v2

CUDA_VISIBLE_DEVICES=0 OCR_V2_PADDLE_DEVICE=gpu:0 OCR_V2_PADDLE_ENGINE=paddle_dynamic OCR_V2_VINTERN_ATTN=flash_attention_2 \
python run_single_frame.py \
  --image /tmp2/maitanha/vgu/ttn/data/AIC2025/keyframes/Keyframes_L27/L27_V010/050.jpg \
  --output_dir ./outputs_test/L27_V010_050 \
  --verbose
```

Test nhanh không chạy Vintern:

```bash
CUDA_VISIBLE_DEVICES=0 OCR_V2_PADDLE_DEVICE=gpu:0 OCR_V2_PADDLE_ENGINE=paddle_dynamic \
python run_single_frame.py \
  --image /tmp2/maitanha/vgu/ttn/data/AIC2025/keyframes/Keyframes_L27/L27_V010/050.jpg \
  --output_dir ./outputs_test/L27_V010_050_no_vintern \
  --no_vintern \
  --verbose
```

### 2.2. Chạy batch một video/folder frame

File:

```text
run_frame_folder.py
```

Dùng khi đã tin pipeline ổn trên vài frame và muốn chạy cả video.

Lệnh khuyến nghị cho `L22_V012`:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/OCR/ocr_vlm_pipeline_v2

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

Ý nghĩa các flag:

| Flag | Ý nghĩa |
|---|---|
| `--workers 4` | Chạy 4 process song song. Mỗi worker load detector, VietOCR, Vintern riêng. |
| `--no_vietocr_batch` | Dùng official VietOCR predict path ổn định hơn. Batch decoder hiện là experimental. |
| `--batch_artifacts minimal` | Chỉ ghi JSONL + summary, không ghi CSV/TXT/PNG từng frame. |
| `--cleanup_crops` | Xóa line/group crops sau mỗi frame để tránh đầy disk. |
| `--quiet` | Giảm log rác từ Paddle/VietOCR/Transformers. |
| `OCR_V2_VINTERN_ATTN=flash_attention_2` | Ưu tiên FlashAttention nếu import được, code sẽ fallback nếu lỗi. |

---

## 3. Runtime environment trên server

Env đang dùng:

```text
/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-detect-gpu
```

Biến môi trường quan trọng:

```bash
CUDA_VISIBLE_DEVICES=0
OCR_V2_PADDLE_DEVICE=gpu:0
OCR_V2_PADDLE_ENGINE=paddle_dynamic
OCR_V2_VINTERN_ATTN=flash_attention_2
```

Ghi chú:

- `CUDA_VISIBLE_DEVICES=0` chọn GPU vật lý.
- Bên trong process, Paddle/Torch thấy GPU đó là `gpu:0` hoặc `cuda:0`.
- Nếu đổi sang GPU 7 thì dùng `CUDA_VISIBLE_DEVICES=7`, vẫn giữ `OCR_V2_PADDLE_DEVICE=gpu:0`.
- `paddle_dynamic` đang được dùng để tránh một số lỗi cuDNN/static trên server.
- Nếu `flash_attention_2` không khả dụng hoặc ABI lỗi, Vintern loader sẽ fallback theo thứ tự `sdpa`, rồi `eager`.

---

## 4. Workflow chi tiết của 1 frame

### 4.1. Load image và wordlist

`run_ocr_pipeline()` đọc ảnh bằng OpenCV:

```text
image_path -> cv2.imread -> BGR -> RGB
```

Wordlist được load từ:

```text
word_list/vn_dictionary.txt
word_list/general_dict.txt
```

Wordlist dùng để tính:

- `lex_ratio`
- token OOV
- nghi lỗi dấu tiếng Việt
- composite score

Nếu wordlist không đủ lớn, pipeline vẫn chạy nhưng dùng `neutral_lex_ratio_if_no_wordlist = 0.50`.

### 4.2. PP-OCRv6 detection

Module:

```text
ocr_pipeline/detector.py
```

Model chính:

```text
PP-OCRv6_medium_det
```

Config mặc định:

```python
det_limit_side_len = 960
det_thresh = 0.30
det_box_thresh = 0.50
det_unclip_ratio = 1.8
```

Detector loader thử nhiều API để tương thích version:

```text
paddleocr.TextDetection
-> paddlex.TextDetection
-> PaddleOCR end-to-end detector
-> paddlex.create_model
```

Output detector được parse từ nhiều field có thể gặp:

```text
dt_polys
rec_polys
boxes
polys
dt_scores
scores
```

### 4.3. Filter box detect

Pipeline lọc raw boxes theo:

```text
det_score
min_box_width
min_box_height
min_box_aspect_ratio
```

Điểm fix quan trọng so với bản cũ:

```text
box_aspect = width / height
```

Không dùng:

```text
min(width, height) / max(width, height)
```

Lý do: subtitle/ticker thường rất dài và mỏng. Nếu dùng `min/max`, các box này bị drop nhầm dù detector đã detect đúng.

Mỗi line item giữ thêm:

```text
box_width
box_height
box_aspect
box_thinness
```

Log debug có thể cho biết số box bị drop theo reason:

```text
After filtering: N valid lines (dropped M, reasons={...})
```

### 4.4. Perspective line crop

Module:

```text
ocr_pipeline/cropping.py
```

Với mỗi detected line:

```text
polygon box
-> perspective crop
-> resize/upscale nếu quá thấp
-> tăng contrast
-> thêm white border
-> save vào ppocr_line_perspective_crops/
```

Config:

```python
perspective_padding = 12
crop_min_height_for_ocr = 72
crop_upscale_max_factor = 3.5
add_white_border = 10
contrast_factor = 1.25
```

Line crop là input chính của VietOCR và Vintern line fallback.

### 4.5. Grouping line boxes

Module:

```text
ocr_pipeline/grouping.py
```

Mục tiêu: gộp các line gần nhau thành một text block để Vintern có context.

Hai line được group nếu:

```text
x_overlap_ratio >= group_min_x_overlap_ratio
vertical_gap <= avg_line_height * group_max_vertical_gap_ratio
```

Config:

```python
group_min_x_overlap_ratio = 0.12
group_max_vertical_gap_ratio = 1.80
group_min_lines = 2
group_crop_padding = 28
```

Implementation dùng Union-Find. Mỗi group có:

```text
group_id
reading_order
line_indices
num_lines
is_multiline_group
bbox_xyxy
group_crop_path
```

Group crop là input của Vintern group fallback.

### 4.6. VietOCR recognition

Module:

```text
ocr_pipeline/recognizers/vietocr_recognizer.py
```

Config chính:

```python
vietocr_config = "vgg_transformer"
vietocr_beamsearch = False
vietocr_use_batch = False
vietocr_batch_size = 16
use_vietocr_return_prob = True
```

Mặc định hiện tại dùng official `Predictor.predict()` từng crop vì ổn định hơn.

Pipeline có hỗ trợ experimental batch path:

```text
batch CNN encode
-> sequential greedy decode
-> detect suspicious output
-> fallback về official Predictor nếu output nhiễu
```

Tuy nhiên khi chạy thật nên giữ:

```bash
--no_vietocr_batch
```

vì batch decoder từng tạo các chuỗi nhiễu kiểu:

```text
AIIIAAIAIIII...
1B0CBC2BBC...
NN1(N1N...
```

### 4.7. rec_conf flat/missing detection

VietOCR không phải lúc nào cũng trả confidence thật. Pipeline xử lý nhiều output format:

```text
(text, prob) -> return_prob_valid
[text, prob] -> return_prob_list_valid
text only -> return_prob_no_conf
missing -> return_prob_missing
error -> error
```

Nếu confidence không có:

```python
rec_conf_fallback_when_missing = 0.50
```

Sau khi chạy toàn bộ line, pipeline kiểm tra `rec_conf_flat`:

```text
std(rec_conf) <= 0.02
hoặc unique_ratio <= 0.10
```

Nếu flat/missing, gating sẽ chuyển sang rule thận trọng hơn.

### 4.8. Structural filter và noise filter

Module:

```text
ocr_pipeline/structural_filter.py
```

Filter loại bỏ trước khi index:

- empty text
- timestamp
- logo/watermark góc phải trên
- bottom counter
- biến thể `60 giây`
- text nhiễu dài ít diversity
- chuỗi nhiều symbol
- chuỗi không có vowel bất thường

Noise filter cũng được chạy sau khi có `final_text` để chặn hallucination từ OCR/VLM.

Ví dụ noise bị chặn:

```text
AIIIAAIAIIIIIIIAAAA...
1B0CBC2BBCBGCBGG...
NN1(N1N([N...
```

### 4.9. Scoring và gating

Module:

```text
ocr_pipeline/scoring/gating.py
ocr_pipeline/scoring/features.py
ocr_pipeline/scoring/wordlist.py
```

Mỗi line được tính:

```text
tokens
eval_tokens
lex_ratio
diacritic_susp
oov_tokens
charset_penalty
repetition_penalty
det_score_norm
composite_score
quality_score
priority
```

Composite score mặc định:

```python
score =
  0.42 * rec_conf
  + 0.18 * det_score
  + 0.22 * lex_ratio
  - 0.12 * diacritic_susp
  - 0.06 * charset_penalty
  - 0.08 * repetition_penalty
```

Gating statuses:

| Status | Ý nghĩa |
|---|---|
| `auto_accept` | VietOCR đủ tin cậy, đưa vào clean text. |
| `vlm_candidate` | Nghi ngờ, gửi sang Vintern nếu nằm trong top candidates. |
| `structural_filter` | Bỏ qua, không index, không review. |
| `review_only` | Có text nhưng vẫn cần review. |
| `text_noise_filter` | Bị loại sau khi phát hiện noise/hallucination. |

Với `rec_conf` đáng tin, auto-accept cần:

```text
rec_conf >= 0.90
det_score >= 0.70
lex_ratio >= 0.50
diacritic_susp <= 0.00
charset_penalty <= 0.05
```

Với `rec_conf` missing/flat, auto-accept thận trọng hơn:

```text
token_count >= 5
det_score >= 0.78
lex_ratio >= 0.70
diacritic_susp <= 0.05
charset_penalty <= 0.02
```

### 4.10. Vintern VLM fallback

Module:

```text
ocr_pipeline/recognizers/vintern_recognizer.py
```

Model mặc định hiện tại:

```text
5CD-AI/Vintern-1B-v3_5
```

Config:

```python
vintern_quantization = "4bit_nf4"
vintern_input_size = 448
vintern_max_tiles = 4
vintern_max_new_tokens = 256
vintern_do_sample = False
```

Vintern nhận input là:

```text
image crop + prompt
```

Có 2 loại crop:

| Loại input | Nguồn | Khi dùng |
|---|---|---|
| Line crop | `ppocr_line_perspective_crops/` | Một dòng bị nghi ngờ. |
| Group crop | `ppocr_stacked_group_crops/` | Cụm nhiều dòng cần context. |

Vintern không tự detect text box. Nó chỉ OCR vùng crop đã đưa vào.

Line prompt:

```text
Hãy đọc chính xác toàn bộ chữ trong ảnh crop này.
Chỉ trả về nội dung OCR, không giải thích.
Giữ nguyên tiếng Việt có dấu nếu có.
```

Group prompt:

```text
Hãy đọc chính xác toàn bộ chữ trong ảnh crop này theo đúng từng dòng.
Nếu ảnh có nhiều dòng chữ, hãy xuống dòng giữa các dòng.
Chỉ trả về nội dung OCR, không giải thích.
```

Preprocess Vintern:

```text
PIL image
-> dynamic_preprocess
-> resize/split thành tile 448x448
-> normalize ImageNet mean/std
-> tensor [num_tiles, 3, 448, 448]
-> model.chat(tokenizer, pixel_values, "<image>\n<prompt>", generation_config)
```

Attention implementation:

```text
env/config explicit
-> flash_attention_2 nếu import được
-> sdpa nếu PyTorch hỗ trợ
-> eager cuối cùng
```

Line candidates:

```text
line.send_to_vintern == True
sort theo priority giảm dần
take top vintern_max_candidates, mặc định 8
```

Group candidates:

```text
group có num_lines >= vintern_group_min_lines
và có ít nhất một line need_review hoặc send_to_vintern
sort theo group_vlm_priority
take top vintern_group_max_candidates, mặc định 4
```

### 4.11. Quyết định accept Vintern

Line Vintern:

- Tính composite score cho text Vintern.
- Tính similarity giữa VietOCR text và Vintern text.
- Nếu agreement cao, chọn text có score tốt hơn và `need_review=False`.
- Nếu disagreement nhưng Vintern tốt hơn đủ margin, override và `need_review=True`.
- Nếu Vintern yếu/noise, giữ VietOCR nhưng không index, đưa review.

Ngưỡng chính:

```python
vintern_min_composite_accept = 0.55
vintern_override_margin = 0.05
vintern_agreement_threshold = 0.82
```

Group Vintern:

- Chạy trên group crop.
- Tính composite score và similarity với review text cũ.
- Accept nếu composite đủ cao hoặc pass soft rule độ dài/lex/charset.

Ngưỡng chính:

```python
group_vintern_min_composite_accept = 0.55
group_vintern_accept_lex_ratio = 0.45
group_vintern_accept_max_charset = 0.05
group_vintern_agreement_threshold = 0.82
```

Nếu group Vintern được accept:

```text
group_text_clean = group_vintern_text
group_text = group_vintern_text
group_final_source = "vintern_group_fallback"
group_keep_for_index = True
```

---

## 5. Text aggregation rule

Module:

```text
ocr_pipeline/outputs.py
```

Rule bắt buộc:

```text
group_text = group_text_clean
```

Không được làm:

```text
group_text = clean_text if clean_text else review_text
```

Lý do: review text chưa được accept không được lọt vào index chính.

Mỗi group có:

```text
group_text_clean
group_text_review
group_text
group_text_lines
group_review_lines
num_keep_lines
num_filtered_lines
num_vlm_candidates
num_vintern_lines
```

Final frame text:

```text
final_clean_text = join(group_text_clean)
final_review_text = join(group_text_review)
```

---

## 6. Output của single-frame mode

Khi chạy `run_single_frame.py`, mặc định ghi đầy đủ artifact:

```text
<frame>_ocr_lines_wordlist_gating_v2.csv
<frame>_ocr_groups_wordlist_gating_v2.csv
<frame>_ocr_clean_text_wordlist_gating_v2.txt
<frame>_ocr_review_text_wordlist_gating_v2.txt
<frame>_ocr_es_doc_wordlist_gating_v2.json
<frame>_ocr_vis_wordlist_gating_v2.png
ppocr_line_perspective_crops/
ppocr_stacked_group_crops/
```

Mục đích:

- CSV line-level để xem từng box.
- CSV group-level để xem group text.
- TXT clean/review để xem text cuối.
- JSON ES document để kiểm tra schema.
- PNG visualization để xem detect/group/filter trên ảnh.
- Crops để kiểm tra input thật của VietOCR/Vintern.

---

## 7. Output của batch video mode

Khi chạy `run_frame_folder.py --batch_artifacts minimal`, output chính:

```text
outputs_video/<video_id>/
├── <video_id>_ocr_es_docs.jsonl
├── <video_id>_ocr_frame_summary.csv
├── <video_id>_ocr_timing_summary.json
└── frames/
```

Với `--cleanup_crops`, crop folders trong từng frame sẽ bị xóa sau khi JSONL đã được ghi.

Không nên bật `--batch_artifacts full` cho dữ liệu lớn, vì sẽ tạo quá nhiều:

```text
CSV
TXT
PNG
JSON từng frame
line crops
group crops
```

---

## 8. ES-friendly schema

Mỗi dòng trong JSONL là một document cho một frame.

Các field chính:

```text
schema_version
document_id
video_id
frame_id
frame_number
image_path
media
ocr_pipeline
timing
quality
ocr_text_clean
ocr_text_review
ocr_text_search
ocr_text_unaccent
ocr_terms
ocr_group_texts_clean
ocr_group_texts_review
ocr_lines
ocr_groups
```

Search field:

| Field | Vai trò |
|---|---|
| `ocr_text_search` | Text clean đã normalize space, dùng search chính. |
| `ocr_text_unaccent` | Bản không dấu, casefold, hỗ trợ fuzzy/lazy search tiếng Việt. |
| `ocr_terms` | Token unique để autocomplete/filter đơn giản. |
| `ocr_text_review` | Text nghi ngờ, không search chính trừ khi bật option. |

Mặc định:

```python
es_include_review_in_search = False
```

Tức là review text không vào `ocr_text_search`.

Quality field có:

```text
lines_detected
groups_formed
clean_chars
review_chars
num_keep_lines
num_review_lines
num_noise_filtered_lines
num_vintern_lines
num_vlm_candidates
gating_stats
final_source_stats
```

---

## 9. Search local trước khi đưa vào ES

File:

```text
search_ocr_results.py
```

Dùng để kiểm tra JSONL output mà chưa cần Elasticsearch.

Ví dụ:

```bash
python search_ocr_results.py \
  --jsonl outputs_video/L22_V012_full_w4 \
  --query "khoa nhiễm thần kinh" \
  --top_k 10 \
  --show_lines
```

Search có hỗ trợ:

- đọc một file JSONL
- đọc folder output chứa `*_ocr_es_docs.jsonl`
- normalize tiếng Việt không dấu
- fuzzy token matching bằng `difflib.SequenceMatcher`
- search review text nếu thêm `--include_review`
- lọc frame range bằng `--frame_from`, `--frame_to`
- xuất JSON bằng `--json`

Ví dụ search cả review text:

```bash
python search_ocr_results.py \
  --jsonl outputs_video/L22_V012_full_w4 \
  --query "my phuoc tan van" \
  --include_review \
  --show_lines
```

---

## 10. Performance và scale

### 10.1. Progress hiện tại

Batch runner hiển thị progress frame-level:

```text
OCR L22_V012_full_w4: 100%| 283/283 [05:29<00:00, 1.16s/frame, avg=1.16s ETA=0s last=227]
```

Thông tin cần theo dõi:

```text
processed / total
elapsed
ETA
sec/frame
avg sec/frame
last frame id
```

Nếu không dùng tqdm, logger vẫn in progress message có:

```text
det time
vietocr time
vintern time
clean chars
review chars
projected total
```

### 10.2. Multi-worker

`run_frame_folder.py` hỗ trợ:

```bash
--workers N
```

Với `workers > 1`, code dùng `ProcessPoolExecutor` với multiprocessing `spawn`.

Mỗi worker tự load:

```text
wordlist
detector
VietOCR predictor
Vintern model nếu bật
```

Điều này tăng throughput nhưng cũng nhân VRAM/model memory theo số worker.

Kinh nghiệm hiện tại trên A5000 24GB:

```text
workers=4
GPU util có thể đạt 100%
VRAM khoảng 8-12GB tùy workload
```

Không phải càng nhiều worker càng nhanh. Nếu `GPU-Util` đã 95-100%, pipeline đang compute-bound. Tăng worker thêm có thể làm chậm vì tranh GPU và tăng overhead.

### 10.3. Ước tính 1M frame

Từ log thực tế:

```text
283 frames / 05:29 ≈ 1.16 sec/frame
```

Ước tính:

```text
1,000,000 frames × 1.16 sec ≈ 1,160,000 sec
≈ 322 giờ
≈ 13.4 ngày / 1 GPU
```

Nên dùng biên an toàn:

```text
1 GPU  -> 14-17 ngày
2 GPU  -> 7-8.5 ngày
4 GPU  -> 3.5-4.2 ngày
8 GPU  -> 1.8-2.1 ngày
```

Nếu video có nhiều text hơn hoặc nhiều Vintern fallback hơn, thời gian sẽ tăng.

### 10.4. Storage strategy cho dữ liệu lớn

Với dữ liệu lớn, luôn chạy:

```bash
--batch_artifacts minimal --cleanup_crops --quiet
```

Chỉ giữ:

```text
ES JSONL
frame summary CSV
timing summary JSON
command/config log nếu cần
```

Không giữ crop/visualization đại trà cho 1M frame.

Chỉ tạo artifact debug cho:

```text
sample nhỏ
frame lỗi
frame có review_chars cao
frame có low confidence
frame cần audit thủ công
```

---

## 11. Checklist chạy thật

### 11.1. Trước khi chạy

Kiểm tra GPU/disk:

```bash
nvidia-smi
df -h /tmp2
du -sh outputs_video 2>/dev/null
```

Kiểm tra env:

```bash
which python
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import paddle; print('paddle', paddle.__version__, paddle.is_compiled_with_cuda())"
python -c "import paddleocr; print('paddleocr ok')"
```

### 11.2. Smoke test

Chạy 20 frame:

```bash
CUDA_VISIBLE_DEVICES=0 OCR_V2_PADDLE_DEVICE=gpu:0 OCR_V2_PADDLE_ENGINE=paddle_dynamic OCR_V2_VINTERN_ATTN=flash_attention_2 \
python run_frame_folder.py \
  --frames_dir ../../../keyframe_test/L22_V012 \
  --output_dir ./outputs_video \
  --video_id L22_V012_smoke_w2 \
  --limit 20 \
  --workers 2 \
  --no_vietocr_batch \
  --batch_artifacts minimal \
  --cleanup_crops \
  --quiet
```

Chạy 40 frame với worker cao hơn:

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

Sau đó search thử:

```bash
python search_ocr_results.py \
  --jsonl outputs_video/L22_V012_smoke_w4 \
  --query "khoa nhiễm thần kinh" \
  --show_lines
```

### 11.3. Chạy full video

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

---

## 12. Những điểm cần tránh

Không chạy full dataset với:

```bash
--batch_artifacts full
```

trừ khi chỉ test rất ít frame.

Không bật VietOCR batch decoder cho run chính nếu chưa kiểm tra kỹ:

```bash
--vietocr_batch
```

vì batch path hiện là experimental và có fallback nhưng vẫn cần audit.

Không index review text vào search chính nếu chưa có lý do rõ ràng:

```python
es_include_review_in_search = False
```

Không tăng `--workers` chỉ vì VRAM còn trống. Nếu GPU util đã 100%, worker cao hơn thường không giúp.

Không đánh giá pipeline chỉ bằng `clean_text`; cần xem thêm:

```text
review_chars
gating_stats
num_vlm_candidates
num_noise_filtered_lines
visualization trên sample
search local trên JSONL
```

---

## 13. Roadmap gần nhất

Các cải tiến nên ưu tiên tiếp theo:

1. Thêm `--shard_index` và `--num_shards` cho `run_frame_folder.py` để chia việc sạch trên nhiều GPU/server.
2. Thêm option xóa empty `frames/<frame_id>/` sau khi `--cleanup_crops`.
3. Thêm mode chỉ lưu crop/visualization cho frame có `review_chars > 0` hoặc `num_vlm_candidates > 0`.
4. Benchmark chính thức `workers=1/2/3/4` trên cùng một video để chọn cấu hình ổn định.
5. Chuẩn hóa mapping Elasticsearch thật dựa trên các field `ocr_text_search`, `ocr_text_unaccent`, `ocr_terms`, `quality`, `timing`.

---

## 14. Tóm tắt cuối

Workflow hiện tại:

```text
PP-OCRv6 detect
-> filter box bằng width/height aspect
-> perspective line crop
-> group line boxes
-> VietOCR đọc line crop
-> wordlist + composite score
-> structural/noise filter
-> Vintern đọc line/group crop nếu cần
-> clean/review aggregation
-> ES-friendly JSONL
-> local fuzzy search hoặc Elasticsearch
```

Triết lý chính:

```text
VietOCR là recognizer nhanh.
Vintern là fallback đắt tiền cho crop nghi ngờ.
Wordlist/gating giảm số crop gửi VLM.
Clean text mới được index.
Review text phục vụ debug/audit.
Batch mode phải tối ưu I/O và storage trước khi scale lên 1M frame.
```
