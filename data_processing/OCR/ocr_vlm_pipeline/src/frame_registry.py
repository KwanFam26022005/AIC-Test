from __future__ import annotations

from pathlib import Path

from .io_utils import ensure_dir, write_table


def frame_number_from_name(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


def build_frame_registry(video_id: str, frames_dir: str | Path, with_phash: bool = False):
    import pandas as pd
    from PIL import Image

    frames_dir = Path(frames_dir)
    rows = []
    for path in sorted(frames_dir.rglob("*.jpg")):
        row = {
            "video_id": video_id,
            "frame_id": path.stem,
            "frame_number": frame_number_from_name(path),
            "frame_path": str(path),
            "width": None,
            "height": None,
            "file_size": path.stat().st_size if path.exists() else 0,
            "phash": None,
            "scene_id": None,
            "is_duplicate": False,
            "source_status": "ok",
        }
        try:
            with Image.open(path) as img:
                row["width"], row["height"] = img.size
                if with_phash:
                    try:
                        import imagehash

                        row["phash"] = str(imagehash.phash(img))
                    except ImportError:
                        row["phash"] = None
        except Exception as exc:
            row["source_status"] = f"error:{type(exc).__name__}"
        rows.append(row)
    return pd.DataFrame(rows)


def run_frame_registry(cfg) -> Path:
    out = Path(cfg.project.output_dir) / "frame_registry.parquet"
    if out.exists() and cfg.project.get("resume", True):
        return out
    ensure_dir(out.parent)
    df = build_frame_registry(
        cfg.project.video_id,
        cfg.project.frames_dir,
        bool(cfg.get("dedup", {}).get("frame_phash", False)),
    )
    return write_table(df, out)

