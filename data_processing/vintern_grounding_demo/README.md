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
  --image /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test/L22_V012/L22_V012_001.jpg \
  --bbox 100,120,420,190 \
  --ocr-text "text loi neu co" \
  --model-id 5CD-AI/Vintern-3B-beta \
  --output-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/vintern_grounding_demo/L22_V012_001
```

Nếu muốn chạy trên GPU khác, đổi `CUDA_VISIBLE_DEVICES`, ví dụ `CUDA_VISIBLE_DEVICES=1`. Bên trong process vẫn dùng `cuda:0` vì CUDA đã remap device.

## Output

Script tạo:

- `result.json`: prompt, response, thời gian, tile count, VRAM peak từng case.
- `full_with_box.jpg`: full frame có vẽ bbox.
- `crop_expanded.jpg`: crop vùng bbox đã mở rộng nhẹ.
- `summary.html`: mở bằng browser/http.server để xem nhanh.

## Cách đọc kết quả

- Nếu `full_coord` sai nhưng `crop_expanded` đúng: Vintern đọc chữ được, nhưng coordinate grounding yếu hoặc bị ảnh hưởng resize/tile.
- Nếu `full_box` đúng hơn `full_coord`: model bám visual marker tốt hơn tọa độ số.
- Nếu cả ba đều sai: vùng chữ quá mờ/nhỏ, bbox sai, hoặc model không đủ OCR năng lực cho case đó.
- Nếu `full_coord` đúng ổn định qua nhiều frame/bbox: có thể cân nhắc full frame + tọa độ, nhưng vẫn cần benchmark thêm.
