# PP-OCRv6 + VLM Correction + Elasticsearch Pipeline

This package refactors the demo notebook into restartable OCR/VLM/text-search stages:

```bash
python scripts/01_build_frame_registry.py --config configs/colab_demo.yaml
python scripts/02_run_ppocr.py --config configs/colab_demo.yaml
python scripts/03_build_vlm_jobs.py --config configs/colab_demo.yaml
python scripts/04_run_vlm_correction.py --config configs/colab_demo.yaml
python scripts/05_merge_features.py --config configs/colab_demo.yaml
python scripts/06_build_es_documents.py --config configs/colab_demo.yaml
```

OCR lines are grouped before VLM correction using spatial connected components:
boxes are linked when they overlap by IoU, are stacked vertically with enough
horizontal overlap, or sit on the same text row with a small horizontal gap. Each
VLM crop is the union of all linked boxes plus `grouping.crop_padding`.

Run tests from this directory:

```bash
python -m pytest tests
```

After stage 5, summarize OCR results:

```bash
python scripts/qa_ocr_stats.py --config configs/server_keyframe_test.yaml
```

This prints OCR/VLM statistics and writes `qa_summary.json`.

Visualize one frame as `PP-OCRv6 detect -> crop -> final`:

```bash
python scripts/visualize_ocr_frame.py --config configs/server_keyframe_test.yaml --frame-id 017
```

This saves a three-panel image under `outputs/<video_id>/qa_frames/`.

Benchmark defaults include:

```text
5CD-AI/Vintern-3B-R-beta
erax-ai/EraX-VL-2B-V1.5
```

Colab environment note: the reference notebook installs `paddlepaddle-gpu` from
Paddle's CUDA-specific index, while VLM inference uses `transformers`,
`accelerate`, and `bitsandbytes`. Run PP-OCR and VLM as separate stages, ideally
in separate Colab runtime sessions, if CUDA package conflicts appear.
