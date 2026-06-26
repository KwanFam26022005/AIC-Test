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

Benchmark defaults include:

```text
5CD-AI/Vintern-3B-R-beta
erax-ai/EraX-VL-2B-V1.5
```

Colab environment note: the reference notebook installs `paddlepaddle-gpu` from
Paddle's CUDA-specific index, while VLM inference uses `transformers`,
`accelerate`, and `bitsandbytes`. Run PP-OCR and VLM as separate stages, ideally
in separate Colab runtime sessions, if CUDA package conflicts appear.
