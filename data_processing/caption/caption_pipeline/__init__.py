"""Caption evidence alignment and indexing pipeline.

Phase 0/1: Align OCR, object detection, and audio evidence per frame,
group into shots, and build a compact search index — all without
any LLM/VLM dependency.
"""

__version__ = "0.1.0"
