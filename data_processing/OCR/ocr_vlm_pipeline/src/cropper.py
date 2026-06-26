from __future__ import annotations

from pathlib import Path

from .io_utils import ensure_dir


def crop_group_images(groups, cfg):
    from PIL import Image

    rows = []
    crop_dir = ensure_dir(Path(cfg.project.output_dir) / "crops")
    padding = int(cfg.grouping.crop_padding)
    min_side = int(cfg.grouping.min_crop_side)
    for row in groups.to_dict("records"):
        if not row.get("need_vlm_group"):
            rows.append(row)
            continue
        try:
            with Image.open(row["frame_path"]) as img:
                width, height = img.size
                x1, y1, x2, y2 = row["merged_bbox"]
                box = (max(0, x1 - padding), max(0, y1 - padding), min(width, x2 + padding), min(height, y2 + padding))
                crop = img.crop(box)
                if min(crop.size) < min_side:
                    scale = min_side / max(1, min(crop.size))
                    crop = crop.resize((int(crop.width * scale), int(crop.height * scale)))
                path = crop_dir / f"{row['frame_id']}_g{row['group_id']:03d}.jpg"
                crop.save(path, quality=95)
                row["crop_path"] = str(path)
                try:
                    import imagehash

                    row["crop_phash"] = str(imagehash.phash(crop))
                except ImportError:
                    row["crop_phash"] = None
        except Exception as exc:
            row["crop_error"] = type(exc).__name__
        rows.append(row)
    import pandas as pd

    return pd.DataFrame(rows)

