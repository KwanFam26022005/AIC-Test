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

Run tests from this directory:

```bash
python -m pytest tests
```

After stage 5, summarize and visualize OCR results:

```bash
python scripts/qa_ocr_results.py --config configs/server_keyframe_test.yaml --frame-id 017
```

This prints OCR/VLM statistics, writes `qa_summary.json`, and saves an annotated
frame under `outputs/<video_id>/qa_frames/`.

Benchmark defaults include:

```text
5CD-AI/Vintern-3B-R-beta
erax-ai/EraX-VL-2B-V1.5
```

Colab environment note: the reference notebook installs `paddlepaddle-gpu` from
Paddle's CUDA-specific index, while VLM inference uses `transformers`,
`accelerate`, and `bitsandbytes`. Run PP-OCR and VLM as separate stages, ideally
in separate Colab runtime sessions, if CUDA package conflicts appear.
