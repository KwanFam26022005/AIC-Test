# -*- coding: utf-8 -*-
"""
tag_canonicalization.py — Xử lý hierarchy collision cho RAM++ + GroundingDINO

Pipeline:
    RAM++ tags → [Layer 1] filter → GroundingDINO → [Layer 2] NMS → [Layer 3] normalize → output

3 Layers:
    Layer 1: filter_tags_for_dino()  — Lọc tags trước khi gửi prompt (best-effort)
    Layer 2: class_agnostic_nms()    — Loại bbox trùng sau detect (PRIMARY defense)
    Layer 3: normalize_label()       — Chuẩn hóa label + loại bbox scene-level

Thiết kế:
    - Layer 2 (NMS) là phòng tuyến CHÍNH — hoạt động trên bbox IoU,
      không phụ thuộc label name → bắt 99% duplicates bất kể RAM++ output gì.
    - Layer 1 (tag filter) là OPTIMIZATION — giảm prompt length, tăng tốc GDINO.
      Chỉ cover high-frequency collisions, KHÔNG cần exhaustive.
    - Layer 3 (normalize) chỉ xử lý edge cases từ GDINO token decoding.

Sử dụng:
    from tag_canonicalization import (
        filter_tags_for_dino,
        class_agnostic_nms,
        normalize_and_filter,
        build_dino_prompt,
    )
"""


# ============================================================================
# LAYER 1 — Tag Pre-filter (best-effort, curated high-frequency collisions)
# ============================================================================

# Synonym map: tag → canonical form
# Chỉ các cặp ĐÃ GẶP trong production data. Thêm khi phát hiện collision mới.
SYNONYM_MAP = {
    # Animal synonyms
    "cattle": "cow",
    "bull": "cow",
    "calf": "cow",
    "heifer": "cow",
    "carp": "fish",
    "salmon": "fish",
    "trout": "fish",
    "tuna": "fish",
    "kitten": "cat",
    "puppy": "dog",
    "pup": "dog",
    "rooster": "chicken",
    "hen": "chicken",
    "mare": "horse",
    "stallion": "horse",
    "pony": "horse",
    "lamb": "sheep",
    "ram": "sheep",
    "goose": "duck",
    # City/scene synonyms
    "city view": "city",
    "city skyline": "city",
    "skyline": "city",
    "night view": "night",
    # Building synonyms
    "hut": "house",
    "house exterior": "house",
    "cottage": "house",
    "cabin": "house",
    # Water synonyms
    "flood water": "flood",
    "sea water": "sea",
    "ocean": "sea",
    # Clothing synonyms
    "mitten": "glove",
    "sneaker": "shoe",
    "boot": "shoe",
    "sandal": "shoe",
    "cowboy hat": "hat",
    "sun hat": "hat",
    "baseball hat": "hat",
    "baseball cap": "hat",
    "cap": "hat",
    "beanie": "hat",
    # Vehicle synonyms
    "van": "car",
    "suv": "car",
    "sedan": "car",
    "automobile": "car",
    "motorbike": "motorcycle",
    "scooter": "motorcycle",
    "bike": "bicycle",
    # Medical synonyms
    "surgeon": "doctor",
    "dentist": "doctor",
    "operating room": "hospital",
    "operating_room": "hospital",
    "hospital room": "hospital",
    "emergency room": "hospital",
    "ward": "hospital",
    "clinic": "hospital",
}

# Hypernym chains: generic_tag → list of specific tags
# Nếu có specific tag → loại generic tag khỏi prompt (giảm duplicate detections)
HYPERNYM_CHAINS = {
    "animal": [
        "cow", "horse", "dog", "cat", "fish", "bird", "sheep", "goat",
        "elephant", "giraffe", "zebra", "bear", "deer", "pig", "rabbit",
        "chicken", "duck", "monkey", "lion", "tiger", "dolphin", "whale",
        "camel", "donkey", "buffalo",
    ],
    "vehicle": [
        "car", "bus", "truck", "motorcycle", "bicycle", "boat", "train",
        "airplane", "helicopter", "ship",
    ],
    "food": [
        "fish", "meat", "bread", "fruit", "vegetable", "cake", "pizza",
        "rice", "noodle", "soup", "salad", "sandwich",
    ],
    "furniture": [
        "chair", "table", "bed", "sofa", "desk", "shelf", "cabinet",
        "stool", "bench", "couch",
    ],
    "clothing": [
        "shirt", "dress", "hat", "shoe", "glove", "jacket", "coat",
        "pants", "skirt", "suit", "uniform",
    ],
    "plant": [
        "tree", "flower", "grass", "bush", "palm",
    ],
}

# Tags KHÔNG cần detect bbox (scene descriptors, actions, colors, media)
# Vẫn được giữ trong output `tags` field, chỉ không gửi cho GDINO.
NON_OBJECT_TAGS = frozenset({
    # Scene descriptors
    "area", "surround", "lush", "grassy", "open", "outdoor", "indoor",
    "rural", "urban", "aerial", "panoramic", "close up", "background",
    "foreground", "landscape", "scenery", "horizon", "scene",
    # Actions / verbs
    "catch", "push", "stand", "stare", "swim", "graze", "wash", "scrub",
    "wear", "walk", "run", "sit", "play", "drive", "fly", "ride",
    "eat", "drink", "cook", "read", "write", "talk", "sleep", "dance",
    "sing", "jump", "climb", "throw", "hold", "carry", "pull",
    "load", "surround",
    # Colors
    "red", "blue", "white", "green", "black", "yellow", "orange",
    "purple", "pink", "brown", "gray", "grey", "golden", "silver",
    # Media / abstract
    "video", "news", "interview", "image", "photo", "film", "show",
    "broadcast", "channel", "program", "television",
    # Qualities / adjectives
    "large", "small", "big", "tall", "short", "long", "wide", "narrow",
    "old", "new", "young", "beautiful", "dirty", "clean", "wet", "dry",
    # Medical abstract terms
    "operate", "operation", "surgery", "perform", "job",
    "procedure", "treatment", "therapy", "diagnosis",
})


def filter_tags_for_dino(raw_tags):
    """Lọc tags từ RAM++ trước khi gửi làm prompt cho GroundingDINO.

    Mục đích: Giảm prompt length → GDINO inference nhanh hơn + giảm duplicates.
    Đây là best-effort optimization, KHÔNG cần cover 100%.
    Class-agnostic NMS (Layer 2) sẽ bắt phần còn lại.

    Args:
        raw_tags: list[str] — tags gốc từ RAM++

    Returns:
        list[str]: tags đã lọc, sẵn sàng cho build_dino_prompt()
    """
    # Step 1: Lowercase + strip + deduplicate
    tags = []
    seen = set()
    for t in raw_tags:
        t_clean = t.lower().strip()
        if t_clean and t_clean not in seen:
            tags.append(t_clean)
            seen.add(t_clean)

    # Step 2: Loại non-object tags (scene/action/color)
    tags = [t for t in tags if t not in NON_OBJECT_TAGS]

    # Step 3: Synonym normalization
    normalized = []
    seen_canonical = set()
    for tag in tags:
        canonical = SYNONYM_MAP.get(tag, tag)
        if canonical not in seen_canonical:
            normalized.append(canonical)
            seen_canonical.add(canonical)
    tags = normalized

    # Step 4: Hypernym removal
    tag_set = set(tags)
    filtered = []
    for tag in tags:
        if tag in HYPERNYM_CHAINS:
            # Nếu có bất kỳ hyponym nào trong tag list → bỏ hypernym
            hyponyms = HYPERNYM_CHAINS[tag]
            if any(h in tag_set for h in hyponyms):
                continue
        filtered.append(tag)

    return filtered


# ============================================================================
# DINO PROMPT BUILDER
# ============================================================================

def build_dino_prompt(tag_list):
    """Chuyển danh sách tags thành text prompt cho GroundingDINO.

    Quy ước chuẩn: chuỗi lowercase, các phrase cách nhau bằng ". "
    và KẾT THÚC bằng dấu chấm.

    Ví dụ: ["person", "cow", "hat"] → "person. cow. hat."

    Args:
        tag_list: list[str] — tags đã filter

    Returns:
        str: text prompt cho GroundingDINO processor
    """
    phrases = [tag.lower().strip() for tag in tag_list if tag.strip()]
    if not phrases:
        return ""
    return ". ".join(phrases) + "."


# ============================================================================
# LAYER 2 — Class-Agnostic NMS (PRIMARY defense — catches ALL duplicates)
# ============================================================================

def compute_iou(box_a, box_b):
    """Tính IoU (Intersection over Union) giữa 2 bbox [x1, y1, x2, y2].

    Args:
        box_a, box_b: list/tuple [x1, y1, x2, y2]

    Returns:
        float: IoU ∈ [0, 1]
    """
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    intersection = (x1 - x0) * (y1 - y0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def class_agnostic_nms(detections, iou_threshold=0.7):
    """Loại bbox trùng GIỮA các class khác nhau (cross-class NMS).

    Đây là phòng tuyến CHÍNH chống hierarchy collision.
    Hoạt động hoàn toàn dựa trên bbox geometry — không phụ thuộc label name.
    → An toàn với mọi tag open-vocab từ RAM++.

    Algorithm: Greedy NMS sorted by score descending.
    Complexity: O(n²) nhưng n thường < 50 detections/frame → negligible.

    Args:
        detections: list[dict] — mỗi dict = {"label", "score", "box"}
        iou_threshold: float — ngưỡng IoU để coi là trùng (default 0.7)

    Returns:
        list[dict]: detections đã loại bỏ duplicates
    """
    if not detections:
        return []

    # Sort by score descending — giữ detection có confidence cao nhất
    sorted_dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []

    for det in sorted_dets:
        is_duplicate = False
        for existing in kept:
            iou = compute_iou(det["box"], existing["box"])
            if iou > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(det)

    return kept


# ============================================================================
# LAYER 3 — Label Normalize + Scene-level Filter
# ============================================================================

# Fix label ghép lỗi từ GroundingDINO token decoding
# Khi 2 tags liên tiếp chia sẻ từ chung, GDINO decode thành label ghép sai
LABEL_NORMALIZE_MAP = {
    "city city view": "city_view",
    "city skyline": "city_skyline",
    "sea water": "sea",
    "house hut": "house",
    "person man": "person",
    "person woman": "person",
    "fish market market": "fish_market",
    "flood water": "flood",
}

# Ngưỡng diện tích tối đa cho bbox (tỷ lệ so với ảnh)
# Bbox chiếm > threshold → loại (scene-level, không phải object riêng lẻ)
SCENE_AREA_THRESHOLD = 0.6


def normalize_label(label):
    """Chuẩn hóa label name: fix ghép lỗi + uppercase.

    Args:
        label: str — label gốc từ GroundingDINO

    Returns:
        str: label đã chuẩn hóa, UPPERCASE
    """
    label_lower = label.lower().strip()
    canonical = LABEL_NORMALIZE_MAP.get(label_lower, label_lower)
    return canonical.upper().replace(" ", "_")


def compute_area_ratio(bbox, img_width, img_height):
    """Tính tỷ lệ diện tích bbox so với ảnh.

    Args:
        bbox: [x1, y1, x2, y2]
        img_width, img_height: kích thước ảnh (pixels)

    Returns:
        float: tỷ lệ ∈ [0, 1]
    """
    box_area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
    img_area = img_width * img_height
    return box_area / img_area if img_area > 0 else 0.0


def normalize_and_filter(detections, img_width, img_height,
                         scene_threshold=SCENE_AREA_THRESHOLD):
    """Chuẩn hóa labels + loại bbox scene-level.

    Args:
        detections: list[dict] — {"label", "score", "box"}
        img_width, img_height: kích thước ảnh
        scene_threshold: ngưỡng area ratio tối đa

    Returns:
        tuple: (kept_objects, scene_labels)
            - kept_objects: list[dict] — objects hợp lệ, label đã normalize
            - scene_labels: list[str] — labels bị loại (chuyển vào tags)
    """
    kept = []
    scene_labels = []

    for det in detections:
        # Normalize label
        det["label"] = normalize_label(det["label"])

        # Check scene-level
        area_ratio = compute_area_ratio(det["box"], img_width, img_height)
        if area_ratio > scene_threshold:
            scene_labels.append(det["label"].lower())
            continue

        kept.append(det)

    return kept, scene_labels


# ============================================================================
# CONVENIENCE — Full canonicalization pipeline
# ============================================================================

def canonicalize_detections(raw_detections, img_width, img_height,
                           nms_iou_threshold=0.7,
                           scene_area_threshold=SCENE_AREA_THRESHOLD):
    """Chạy toàn bộ Layer 2 + Layer 3 trên raw detections.

    Args:
        raw_detections: list[dict] — output thô từ GroundingDINO
        img_width, img_height: kích thước ảnh
        nms_iou_threshold: IoU threshold cho NMS
        scene_area_threshold: area ratio threshold cho scene filter

    Returns:
        tuple: (final_objects, scene_labels)
    """
    # Layer 2: Class-agnostic NMS
    deduped = class_agnostic_nms(raw_detections, iou_threshold=nms_iou_threshold)

    # Layer 3: Normalize + scene filter
    final_objects, scene_labels = normalize_and_filter(
        deduped, img_width, img_height, scene_threshold=scene_area_threshold
    )

    return final_objects, scene_labels
