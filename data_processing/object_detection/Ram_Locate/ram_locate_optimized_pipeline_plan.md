# RAM++ + LocateAnything Optimized Pipeline Plan

Mục tiêu: xây dựng pipeline detect/count object cho keyframe quy mô lớn, dự kiến chạy trên server giống workflow OCR v2, ưu tiên ổn định cho khoảng 1 triệu frame trên 1 GPU RTX A5000 24GB.

Plan này chỉ là thiết kế và cấu hình vận hành, chưa bao gồm code implementation.

---

## 1. Kết luận cấu hình khuyến nghị

Khuyến nghị dùng kiến trúc 2 phase:

```text
Phase A: RAM++ tag cache
  frame -> RAM++ -> raw_tags -> filtered_object_tags -> tags JSONL

Phase B: LocateAnything detection/count
  frame + filtered_object_tags -> LocateAnything -> raw boxes
  -> postprocess -> final objects/counts -> detections JSONL
```

Lý do chọn 2 phase:

- RAM++ và LocateAnything có dependency/runtime khác nhau; tách phase giúp tránh xung đột env.
- A5000 có 24GB VRAM, đủ để chạy từng model ổn định hơn là giữ nhiều model lớn cùng lúc.
- Khi chạy 1 triệu frame, resume/checkpoint quan trọng hơn tiết kiệm một lần đọc ảnh.
- Có thể benchmark riêng RAM++ và LocateAnything để biết bottleneck thật.
- Có thể thay runtime LocateAnything từ official Python sang `locate-anything.cpp` mà không đụng lại tag cache.

Runtime khuyến nghị ban đầu:

```text
RAM++: Python/PyTorch, batch tag extraction
LocateAnything: official Python persistent worker, generation_mode=hybrid
Attention: flash_attention_2 nếu khả dụng, fallback sdpa/default
Alternative runtime: locate-anything.cpp persistent worker/C API nếu official Python quá chậm hoặc env khó ổn định
```

Không khuyến nghị:

```text
1. Chạy LocateAnything CLI mỗi frame nếu CLI phải reload model mỗi lần.
2. Chạy mỗi object tag thành một LocateAnything request riêng.
3. Dùng nhiều LocateAnything worker trên cùng 1 A5000 nếu chưa chứng minh không OOM.
4. Lưu annotated image/crop cho toàn bộ 1 triệu frame.
```

---

## 2. Input và output

Input giống batch OCR v2:

```text
/path/to/keyframes/<video_id>/*.jpg
```

Output tối ưu cho production:

```text
outputs_ram_locate/
├── tag_cache/
│   └── <video_id>_ram_tags.jsonl
├── detections/
│   └── <video_id>_objects.jsonl
├── summaries/
│   ├── <video_id>_object_frame_summary.csv
│   └── <video_id>_timing_summary.json
└── debug_samples/
    └── optional annotated images for sampled/problem frames only
```

Output mỗi frame nên giữ schema gần pipeline `ram_gdino_pipeline.py` hiện có:

```json
{
  "frame_id": "L22_V012_001",
  "video_id": "L22_V012",
  "frame_idx": 1,
  "image_size": [1280, 720],
  "tags": ["person", "car", "street"],
  "object_prompt_tags": ["person", "car"],
  "objects": [
    {"label": "PERSON", "box": [97.5, 182.0, 519.5, 718.4], "source": "locateanything"}
  ],
  "object_summary": ["CAR", "PERSON"],
  "object_counts": {"CAR": 2, "PERSON": 1},
  "quality": {
    "num_raw_tags": 18,
    "num_prompt_tags": 6,
    "num_raw_boxes": 5,
    "num_final_boxes": 3,
    "num_removed_duplicates": 2
  },
  "timing": {
    "ram_sec": 0.0,
    "locate_sec": 2.8,
    "postprocess_sec": 0.01
  }
}
```

Ghi chú: LocateAnything thường không trả confidence score như GroundingDINO, nên không thiết kế output phụ thuộc vào `score`.

---

## 3. Phase A - RAM++ tag cache

Mục tiêu của Phase A là tạo candidate object tags cho từng frame.

Config khuyến nghị:

```text
Model: RAM++ Swin-Large
Checkpoint: ram_plus_swin_large_14m.pth
Image size: 384
Precision: fp16/bf16 nếu model ổn định, fallback fp32 nếu có lỗi
Batch size A5000: thử 16, nếu OOM giảm 8, nếu còn dư tăng 32
Workers CPU load image: 4-8
Output: JSONL append/resume theo frame_id
```

Tag filtering:

```text
1. lowercase + strip + deduplicate
2. loại scene/action/color/media tags
3. synonym normalization
4. hypernym removal
5. giới hạn số prompt tags gửi sang LocateAnything
```

Giới hạn prompt tags:

```text
Default max_prompt_tags: 20
Nếu scene đơn giản: 10-15
Nếu video nhiều vật nhỏ/dense: 25-30
Không vượt 30 nếu chưa benchmark
```

Ưu tiên giữ:

```text
person, vehicle, animal, food, furniture, tool, sign, bag, bottle, phone, laptop,
traffic light, bicycle, motorcycle, bus, truck, boat, train, airplane
```

Loại khỏi prompt nhưng vẫn giữ trong `tags`:

```text
scene, city, outdoor, indoor, landscape, news, video, red, blue, big, small,
walking, standing, sitting, wearing, close up, background
```

---

## 4. Phase B - LocateAnything detection/count

Mục tiêu của Phase B là biến `filtered_object_tags` thành bbox và object count.

Prompt template:

```text
Locate all the instances that matches the following description: person</c>car</c>bicycle.
```

Config khuyến nghị cho A5000:

```text
generation_mode: hybrid
max_new_tokens: 1024
temperature: 0
do_sample: false nếu API cho phép
vision_attention: flash_attention_2 nếu khả dụng, fallback sdpa/eager
batch_size: 1 cho lần đầu production
image_max_side: 1280
```

Khi nào chỉnh:

```text
Nếu thiếu object nhỏ:
  image_max_side -> 1536
  max_prompt_tags -> 25
  max_new_tokens -> 2048

Nếu quá chậm:
  image_max_side -> 960 hoặc 1152
  max_prompt_tags -> 12-15
  thử generation_mode=fast trên sample

Nếu output bị truncate:
  max_new_tokens -> 2048

Nếu nhiều duplicate/degenerate tail boxes:
  giữ max_new_tokens thấp hơn
  bật strict degenerate filter
  tăng NMS hoặc repeated-box filter
```

Không nên batch nhiều ảnh ngay từ đầu trên A5000. LocateAnything official có batch runtime, nhưng với 24GB VRAM cần benchmark trước. Production v1 nên dùng 1 persistent GPU worker, stream từng frame, tối ưu I/O và resume trước.

---

## 5. Postprocess để count ổn định

Vì LocateAnything không có confidence score chuẩn, count phải dựa trên geometry và label hygiene.

Postprocess layers:

```text
Layer 1: parse output
  - lấy label + bbox xyxy
  - clamp bbox vào image bounds
  - loại bbox malformed

Layer 2: degenerate filter
  - width <= 3 px hoặc height <= 3 px -> drop
  - area quá nhỏ -> drop
  - x2 <= x1 hoặc y2 <= y1 -> drop
  - bbox lặp exact/near-exact nhiều lần -> giữ 1

Layer 3: label normalize
  - uppercase
  - space -> underscore
  - synonym map: MAN/WOMAN -> PERSON, AUTOMOBILE/VAN/SUV -> CAR, ...

Layer 4: scene-level filter
  - area_ratio > 0.65 -> chuyển label sang tags/scene_labels, không count

Layer 5: class-agnostic NMS
  - IoU threshold default: 0.75
  - nếu overcount nhiều: 0.65-0.70
  - nếu miss object sát nhau: 0.80-0.85

Layer 6: count
  - object_counts = Counter(final_objects.label)
```

Field `score`:

```text
Không giả lập score nếu model không trả score.
Nếu cần ranking, dùng score nội bộ dạng heuristic riêng, ví dụ source_rank/postprocess_rank,
nhưng không gọi là confidence.
```

---

## 6. Server/env plan

Không nên cài vào env OCR v2 đang chạy ổn, vì OCR có Paddle/Vintern riêng và LocateAnything cần version Transformers mới.

Env khuyến nghị:

```text
Conda/venv name: aic-ram-locate-gpu
Python: 3.10 hoặc 3.11
CUDA: theo driver server, ưu tiên PyTorch CUDA 12.x
GPU: RTX A5000 24GB
```

Packages chính:

```text
torch + torchvision CUDA build
transformers==4.57.1
tokenizers==0.22.0
accelerate==1.5.2
timm>=1.0.11
peft==0.12.0
numpy==1.25.0
Pillow==11.1.0
opencv-python-headless==4.11.0.86
decord==0.6.0
recognize-anything
tqdm
```

Optional acceleration:

```text
flash-attn compatible với GPU/driver
LocateAnything batch_utils/kernel_utils từ model repo nếu dùng la_flash
```

Model/cache layout:

```text
models/
├── ram_plus_swin_large_14m.pth
└── LocateAnything-3B/
```

Environment variables:

```text
CUDA_VISIBLE_DEVICES=0
RAM_CHECKPOINT_PATH=/path/to/models/ram_plus_swin_large_14m.pth
HF_HOME=/path/to/hf_cache
TRANSFORMERS_CACHE=/path/to/hf_cache
```

Nếu dùng LocateAnything batch runtime official:

```text
PYTHONPATH=/path/to/models/LocateAnything-3B:$PYTHONPATH
attn=la_flash
vision_attn=flash_attention_2
scheduler=pipeline
```

Nếu flash attention lỗi:

```text
attn=sdpa
vision_attn=sdpa
```

---

## 7. Batch runner design giống OCR v2

Runner nên theo pattern của `run_frame_folder.py` trong OCR v2:

```text
1. collect frames theo natural sort
2. load model/worker một lần
3. resume theo existing frame_id trong output JSONL
4. process từng frame
5. append JSONL ngay sau mỗi frame
6. flush định kỳ
7. ghi summary CSV/timing JSON
8. progress bar có avg sec/frame, ETA, last frame
```

Artifacts mode:

```text
Production:
  batch_artifacts=minimal
  save_jsonl=true
  save_annotated=false
  save_raw_model_answer=false hoặc sample_only

Debug:
  save_annotated=true cho 20-200 frame sample
  save_raw_model_answer=true
```

Resume:

```text
Phase A resume: skip frame_id đã có trong tag_cache JSONL
Phase B resume: skip frame_id đã có trong detections JSONL
Nếu output line corrupt: bỏ qua line lỗi, xử lý lại frame đó
```

Sharding:

```text
Ưu tiên shard theo video_id.
Nếu 1 video quá lớn, thêm shard_index/num_shards theo frame list.
Mỗi GPU/server xử lý một shard riêng, merge JSONL sau.
```

---

## 8. Benchmark protocol bắt buộc trước khi chạy 1M frame

Không chạy thẳng 1 triệu frame. Cần 4 mức benchmark:

```text
Smoke 20 frame:
  kiểm tra env, output schema, visual quality

Benchmark 1,000 frame:
  đo avg_sec_per_frame, p50/p95, VRAM, GPU util

Pilot 10,000 frame:
  kiểm tra resume, disk, memory leak, JSONL integrity

Full run:
  chạy theo shard/video, monitor ETA và lỗi
```

Metrics cần ghi:

```text
Phase A:
  ram_sec/frame
  num_raw_tags
  num_prompt_tags

Phase B:
  locate_sec/frame
  postprocess_sec/frame
  num_raw_boxes
  num_final_boxes
  num_removed_degenerate
  num_removed_nms
  GPU memory peak
  failures per 1,000 frames
```

Thời gian 1M frame:

```text
days = avg_sec_per_frame * 1_000_000 / 86400
```

Cost cloud:

```text
cost = avg_sec_per_frame * 1_000_000 / 3600 * hourly_rate
```

Planning hiện tại cho A5000:

```text
Best case tối ưu: 1-2 s/frame  -> 12-23 ngày
Realistic:        2-5 s/frame  -> 23-58 ngày
Worst case:       5-10 s/frame -> 58-116 ngày
```

---

## 9. Quality audit plan

Sampling audit:

```text
20 frame đầu tiên: xem toàn bộ annotated/debug
200 frame ngẫu nhiên: kiểm tra label/count
200 frame dense/crowded: kiểm tra duplicate/overcount
200 frame low-light/blur: kiểm tra false positive
```

Các lỗi cần phân loại:

```text
RAM_MISS_TAG: RAM++ không sinh tag nên LocateAnything không detect
PROMPT_TOO_BROAD: tag generic gây box scene-level/duplicate
LOCATE_MISS_SMALL: object nhỏ không được detect
LOCATE_DUPLICATE: nhiều bbox cho cùng object
DEGENERATE_TAIL: bbox lặp/malformed cuối output
LABEL_COLLISION: person/man/woman hoặc car/vehicle bị count tách
SCENE_BOX: bbox phủ gần cả ảnh
```

Quyết định pass/fail trước full run:

```text
1. JSONL không corrupt khi resume.
2. Error rate < 0.5% frame.
3. Không có memory leak rõ sau 10,000 frame.
4. Overcount do duplicate ở sample < mức chấp nhận sau NMS.
5. Avg sec/frame đủ để hoàn thành trong ngân sách thời gian.
```

---

## 10. Cấu hình production v1 đề xuất

Phase A:

```text
ram_image_size: 384
ram_batch_size: 16
max_prompt_tags: 20
tag_filter: enabled
tag_cache_output: JSONL
resume: enabled
```

Phase B:

```text
locate_runtime: official_python
generation_mode: hybrid
max_new_tokens: 1024
temperature: 0
image_max_side: 1280
locate_batch_size: 1
worker_count_per_gpu: 1
attn_implementation: flash_attention_2 if available else sdpa/default
```

Postprocess:

```text
min_box_width_px: 4
min_box_height_px: 4
min_box_area_px: 16
scene_area_threshold: 0.65
nms_iou_threshold: 0.75
repeated_box_iou_threshold: 0.98
normalize_labels: enabled
class_agnostic_nms: enabled
```

Artifacts:

```text
batch_artifacts: minimal
save_annotated: false
save_raw_answer: false for production, true for debug sample
flush_every_n_frames: 1-10
timing_summary: enabled
frame_summary_csv: enabled
```

---

## 11. Khi nào đổi sang locate-anything.cpp

Đổi sang `locate-anything.cpp` nếu gặp một trong các vấn đề:

```text
1. Official Python/Transformers env quá khó ổn định trên server.
2. VRAM/KV cache của official Python vượt 24GB ở resolution mong muốn.
3. Throughput official Python thấp hơn kỳ vọng sau khi đã dùng la_flash/sdpa đúng cách.
4. Muốn runtime inference nhẹ hơn, ít phụ thuộc Python hơn.
```

Config nếu dùng `.cpp`:

```text
Model: locate-anything-q8_0.gguf
Runtime: persistent process hoặc C API, không reload model mỗi frame
Mode: hybrid
Threads CPU: benchmark 8/12/16
GPU backend: build CUDA nếu chạy trên A5000
Output: parse JSON detections từ worker/C API
```

Không dùng `.cpp` theo kiểu:

```text
locate-anything-cli detect ... cho từng frame riêng lẻ nếu mỗi lần phải load GGUF lại
```

---

## 12. References

- NVIDIA LocateAnything HF Space: https://huggingface.co/spaces/nvidia/LocateAnything
- NVIDIA LocateAnything-3B model card: https://huggingface.co/nvidia/LocateAnything-3B
- LocateAnything paper: https://arxiv.org/abs/2605.27365
- locate-anything.cpp: https://github.com/mudler/locate-anything.cpp
- RAM++ / recognize-anything: https://github.com/xinyu1205/recognize-anything
