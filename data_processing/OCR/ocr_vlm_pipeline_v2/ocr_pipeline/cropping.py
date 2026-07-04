"""
Perspective and axis-aligned cropping for OCR line/group crops.

Handles:
- Ordering polygon points (TL, TR, BR, BL)
- Perspective transform for line crops
- Upscale/contrast/border preparation for OCR
- Axis-aligned bbox crop for group context
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: Array of shape [4, 2]

    Returns:
        Ordered array [4, 2]
    """
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum: TL has smallest sum, BR has largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Diff: TR has smallest diff (x-y), BL has largest
    d = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def _expand_polygon(pts: np.ndarray, pad: int, img_shape: tuple) -> np.ndarray:
    """Expand polygon outward by padding, clipped to image bounds."""
    H, W = img_shape[:2]
    center = pts.mean(axis=0)
    expanded = np.zeros_like(pts)
    for i in range(4):
        direction = pts[i] - center
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        expanded[i] = pts[i] + direction * pad

    # Clip to image bounds
    expanded[:, 0] = np.clip(expanded[:, 0], 0, W - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, H - 1)
    return expanded


def crop_perspective_line(img_rgb: np.ndarray, box, pad: int = 12) -> np.ndarray:
    """Crop a text line using perspective transform.

    Args:
        img_rgb: Input image (H, W, 3)
        box: Polygon points [4, 2] or list of [x, y]
        pad: Padding around polygon before cropping

    Returns:
        Cropped line image (numpy array, RGB)
    """
    pts = np.array(box, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected box shape (4, 2), got {pts.shape}")

    ordered = order_points(pts)

    # Expand polygon with padding
    if pad > 0:
        ordered = _expand_polygon(ordered, pad, img_rgb.shape)

    tl, tr, br, bl = ordered

    # Compute output width and height
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    out_w = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    out_h = int(max(height_left, height_right))

    if out_w <= 0 or out_h <= 0:
        # Fallback to axis-aligned crop
        xs = pts[:, 0].astype(int)
        ys = pts[:, 1].astype(int)
        x1, x2 = max(0, xs.min()), min(img_rgb.shape[1], xs.max())
        y1, y2 = max(0, ys.min()), min(img_rgb.shape[0], ys.max())
        return img_rgb[y1:y2, x1:x2].copy()

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst)
    crop = cv2.warpPerspective(img_rgb, M, (out_w, out_h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    return crop


def prepare_crop_for_ocr(
    crop: np.ndarray,
    min_height: int = 72,
    max_upscale_factor: float = 3.5,
    border: int = 10,
    contrast_factor: float = 1.25,
) -> np.ndarray:
    """Prepare a line crop for OCR recognition.

    Steps:
    1. Upscale if height < min_height (up to max_upscale_factor)
    2. Increase contrast
    3. Add white border

    Args:
        crop: Input crop (H, W, 3) RGB
        min_height: Minimum height for OCR
        max_upscale_factor: Maximum upscale ratio
        border: White border width in pixels
        contrast_factor: Contrast enhancement factor (1.0 = no change)

    Returns:
        Prepared crop (numpy array, RGB)
    """
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape[:2]

    # 1. Upscale if too small
    if h < min_height and h > 0:
        scale = min(min_height / h, max_upscale_factor)
        new_w = int(w * scale)
        new_h = int(h * scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 2. Contrast enhancement
    if contrast_factor != 1.0:
        crop = _adjust_contrast(crop, contrast_factor)

    # 3. Add white border
    if border > 0:
        crop = cv2.copyMakeBorder(
            crop, border, border, border, border,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

    return crop


def _adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Adjust image contrast around mean."""
    mean = img.mean()
    adjusted = mean + factor * (img.astype(np.float32) - mean)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def crop_axis_from_bbox(
    img_rgb: np.ndarray,
    bbox_xyxy: list[int],
    pad: int = 28,
) -> np.ndarray:
    """Crop a region using axis-aligned bounding box with padding.

    Args:
        img_rgb: Input image (H, W, 3) RGB
        bbox_xyxy: [x1, y1, x2, y2]
        pad: Padding around bbox

    Returns:
        Cropped region (numpy array, RGB)
    """
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)
    return img_rgb[y1:y2, x1:x2].copy()


def save_line_crop(
    crop: np.ndarray,
    output_dir: str | Path,
    line_id: int,
) -> str:
    """Save a line crop to disk.

    Returns:
        Absolute path to saved image.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"line_{line_id:03d}_persp.png"
    # Convert RGB to BGR for cv2 save
    cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return str(path)


def save_group_crop(
    crop: np.ndarray,
    output_dir: str | Path,
    group_id: int,
) -> str:
    """Save a group crop to disk.

    Returns:
        Absolute path to saved image.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"group_{group_id:03d}.png"
    cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return str(path)
