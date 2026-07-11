# VLM Frame Caption Demo

Demo thí nghiệm visual frame caption cho toàn bộ frame của một video, chạy được với hai provider:

- `vintern`: `5CD-AI/Vintern-3B-beta`, chạy trong env OCR/detect.
- `qwen`: `Qwen/Qwen2.5-VL-7B-Instruct`, chạy trong env caption VLM.

Script chỉ sinh caption trực tiếp từ full frame. Nó không dùng OCR boxes, object labels, hay audio.

## Output

Mỗi provider ghi một JSONL:

```json
{
  "video_id": "L22_V012",
  "frame_id": "L22_V012_216",
  "frame_name": "216",
  "image_path": ".../216.jpg",
  "provider": "vintern",
  "model_id": "5CD-AI/Vintern-3B-beta",
  "caption": "...",
  "raw_response": "...",
  "elapsed_seconds": 1.23,
  "num_tiles": 3,
  "warning": ""
}
```

`--resume` mặc định bật: frame nào đã có trong JSONL sẽ được skip.

## Vintern env

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
source /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-detect-gpu/bin/activate

CUDA_VISIBLE_DEVICES=7 \
OCR_V2_VINTERN_ATTN=flash_attention_2 \
PYTHONIOENCODING=utf-8 \
python data_processing/vlm_frame_caption_demo/run_vlm_frame_caption_demo.py \
  --provider vintern \
  --model-id 5CD-AI/Vintern-3B-beta \
  --video-id L22_V012 \
  --frames-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test/L22_V012 \
  --output-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/vlm_frame_caption_demo/vintern3b/L22_V012 \
  --pattern "*.jpg" \
  --max-new-tokens 96 \
  --max-tiles 4 \
  --resume
```

## Qwen env

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
source /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm/bin/activate

CUDA_VISIBLE_DEVICES=7 \
CAPTION_VLM_ATTN=flash_attention_2 \
PYTHONIOENCODING=utf-8 \
python data_processing/vlm_frame_caption_demo/run_vlm_frame_caption_demo.py \
  --provider qwen \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --video-id L22_V012 \
  --frames-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test/L22_V012 \
  --output-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/vlm_frame_caption_demo/qwen25vl7b/L22_V012 \
  --pattern "*.jpg" \
  --max-new-tokens 96 \
  --resume
```

## Chạy test nhanh

Chỉ chạy 10 frame đầu:

```bash
--max-frames 10 --no-resume
```

Chia shard để chạy song song hai GPU:

```bash
# GPU 0
--num-shards 2 --shard-index 0

# GPU 1
--num-shards 2 --shard-index 1
```

## So sánh kết quả

Sau khi chạy xong hai provider, so sánh:

- tốc độ trung bình trong `report.json`
- caption rỗng / quá ngắn / warning
- chất lượng caption trên một số frame tin tức, map, outdoor, text-heavy

