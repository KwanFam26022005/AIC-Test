# Vintern Grounding Demo

Demo nhỏ để kiểm tra giả thuyết: Vintern-3B-beta có thể đọc OCR theo tọa độ trên full frame tốt đến đâu, so với khi dùng bbox được vẽ trực tiếp hoặc crop vùng chữ.

Demo không thay đổi OCR/caption pipeline chính. Nó tái sử dụng loader và preprocess Vintern trong `data_processing/OCR/ocr_vlm_pipeline_v2` để tránh lệch môi trường.

## Chạy trên server

Từ root repo:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
source /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-detect-gpu/bin/activate

CUDA_VISIBLE_DEVICES=0 \
OCR_V2_PADDLE_DEVICE=gpu:0 \
OCR_V2_PADDLE_ENGINE=paddle_dynamic \
OCR_V2_VINTERN_ATTN=flash_attention_2 \
PYTHONIOENCODING=utf-8 \
python data_processing/vintern_grounding_demo/run_vintern_grounding_demo.py \
  --video-id L22_V012 \
  --frame-id L22_V012_001 \
  --ocr-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2 \
  --frames-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --model-id 5CD-AI/Vintern-3B-beta \
  --output-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/vintern_grounding_demo/L22_V012_001
```

Muốn kiểm tra bbox/crop trước khi load model, thêm `--dry-run`:

```bash
PYTHONIOENCODING=utf-8 \
python data_processing/vintern_grounding_demo/run_vintern_grounding_demo.py \
  --video-id L22_V012 \
  --frame-id L22_V012_001 \
  --ocr-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2 \
  --frames-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --output-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/vintern_grounding_demo/L22_V012_001_dry \
  --dry-run
```

Nếu muốn chạy trên GPU khác, đổi `CUDA_VISIBLE_DEVICES`, ví dụ `CUDA_VISIBLE_DEVICES=1`. Bên trong process vẫn dùng `cuda:0` vì CUDA đã remap device.

## Output

Script tạo:

- `result.json`: prompt, response, thời gian, tile count, VRAM peak từng case.
- `full_with_box.jpg`: full frame có vẽ bbox.
- `crop_expanded.jpg`: crop vùng bbox đã mở rộng nhẹ.
- `summary.html`: mở bằng browser/http.server để xem nhanh.

## Cách đọc kết quả

Khi dùng `--frame-id`, script tự đọc `<ocr-root>/<video-id>/<video-id>_ocr_es_docs.jsonl`, tìm đúng frame, rồi chọn một OCR line bbox đại diện. Mặc định nó ưu tiên line từng được đánh dấu `send_to_vintern` hoặc `need_review`; nếu không có thì chọn line có score thấp/text dài/vùng lớn. Muốn đổi candidate trong cùng frame, tăng `--candidate-index 1`, `--candidate-index 2`, ...

Muốn chạy nhiều bbox trong cùng frame:

```bash
# Chạy top 5 OCR boxes, bắt đầu từ candidate 0.
--max-candidates 5

# Chạy toàn bộ OCR boxes trong frame.
--all-candidates
```

Khi chạy nhiều bbox, output sẽ có thêm:

- `candidate_000_<line_id>/`, `candidate_001_<line_id>/`, ...: kết quả từng bbox.
- `result_all.json`: kết quả tổng hợp.
- `summary_all.html`: bảng so sánh tổng hợp.

- Nếu `full_coord` sai nhưng `crop_expanded` đúng: Vintern đọc chữ được, nhưng coordinate grounding yếu hoặc bị ảnh hưởng resize/tile.
- Nếu `full_box` đúng hơn `full_coord`: model bám visual marker tốt hơn tọa độ số.
- Nếu cả ba đều sai: vùng chữ quá mờ/nhỏ, bbox sai, hoặc model không đủ OCR năng lực cho case đó.
- Nếu `full_coord` đúng ổn định qua nhiều frame/bbox: có thể cân nhắc full frame + tọa độ, nhưng vẫn cần benchmark thêm.
