"""Caption pipeline v2 — visual + audio-context caption generation.

Phases:
  0: Preflight and canonical timeline
  1: Shot-aware frame selection
  2: Qwen2.5-VL visual-only captioning
  3: Validation and caption propagation
  4: Read-only ASR alignment by shot
  5: Text ReCap, one structured request per shot
  6: Optional LVLM escalation/refinement
  7: Canonical outputs, checkpoints, and reports
"""

__version__ = "0.1.0"
