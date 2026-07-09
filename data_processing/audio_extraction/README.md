# Audio feature extraction

This folder contains a first-pass audio pipeline that turns one or more videos
into timestamped audio features for later multimodal merge.

## Server paths for the first test

```text
Project root: /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
OCR dir:      /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/OCR
Test video:   /tmp2/maitanha/vgu/ttn/data/AIC2025/videos/Videos_L22/L22_V012.mp4
```

The audio pipeline uses a separate virtual environment at:

```text
/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/.venv_audio
```

This keeps audio dependencies away from OCR, keyframe, and object-detection
environments.

## Install on the A5000 server

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
bash data_processing/audio_extraction/scripts/setup_audio_env.sh
```

Make sure `ffmpeg` and `ffprobe` are available on `PATH`.

## Smoke test L22_V012.mp4

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
CUDA_VISIBLE_DEVICES=0 bash data_processing/audio_extraction/scripts/run_l22_v012_smoke.sh
```

Useful faster smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 bash data_processing/audio_extraction/scripts/run_l22_v012_smoke.sh --model turbo
```

Extraction/VAD only:

```bash
bash data_processing/audio_extraction/scripts/run_l22_v012_smoke.sh --skip_asr
```

Main output:

```text
outputs/audio/L22_V012/features/audio_features.jsonl
```

The default ASR model is `large-v3` with `device=cuda` and
`compute_type=float16`, which is suitable for an RTX A5000. If you want to use
PhoWhisper with `faster-whisper`, provide a local CTranslate2-converted model
directory via `--model /path/to/phowhisper-ct2` or set `asr.model_path` in the
YAML config.
