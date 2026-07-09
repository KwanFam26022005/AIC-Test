# -*- coding: utf-8 -*-
"""
tag_canonicalization.py — Xử lý hierarchy collision cho RAM++ + GroundingDINO

Pipeline:
    RAM++ tags → [Layer 1] filter → GroundingDINO → [Layer 2] NMS → [Layer 3] normalize → output

4 Layers (v1.1):
    Layer 1: filter_tags_for_dino()  — Lọc tags trước khi gửi prompt (best-effort)
    Layer 2: class_agnostic_nms()    — Loại bbox trùng sau detect (PRIMARY defense)
    Layer 2b: family_aware_nms()     — Dedup person-family hierarchy (PERSON/MAN/WOMAN)
    Layer 3: normalize_label()       — Chuẩn hóa label + loại bbox scene-level
    Layer 4: enrich_detection()      — Thêm geometry metadata (position, size_bucket, etc.)

Thiết kế:
    - Layer 2 (NMS) là phòng tuyến CHÍNH — hoạt động trên bbox IoU,
      không phụ thuộc label name → bắt 99% duplicates bất kể RAM++ output gì.
    - Layer 2b (family NMS) xử lý hierarchy collision khi PERSON + MAN/WOMAN overlap.
    - Layer 1 (tag filter) là OPTIMIZATION — giảm prompt length, tăng tốc GDINO.
      Chỉ cover high-frequency collisions, KHÔNG cần exhaustive.
    - Layer 3 (normalize) chỉ xử lý edge cases từ GDINO token decoding.
    - Layer 4 (enrich) thêm metadata cho caption/search downstream.

Sử dụng:
    from tag_canonicalization import (
        filter_tags_for_dino,
        class_agnostic_nms,
        normalize_and_filter,
        build_dino_prompt,
        enrich_detection,
        family_aware_nms,
        canonicalize_detections,
        build_object_counts,
        build_object_text,
        build_important_objects,
        build_search_fields,
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

# --------------------------------------------------------------------------
# SCENE LABELS — Labels nên phân loại thành scene_tags, không phải countable objects
# Bắt cả trường hợp bbox nhỏ hơn scene_area_threshold nhưng label rõ ràng là scene.
# --------------------------------------------------------------------------
SCENE_LABELS = frozenset({
    "SKY", "CITY", "CITY_SKYLINE", "CITY_VIEW", "WATER", "SEA", "SUN",
    "SUNSET", "SUNRISE", "NIGHT", "NIGHT_VIEW", "ROAD", "LANDSCAPE",
    "BACKGROUND", "OCEAN", "RIVER", "LAKE", "MOUNTAIN", "FIELD",
    "FOREST", "BEACH", "DESERT", "SNOW", "RAIN", "FOG", "CLOUD",
    "SKYLINE", "HORIZON",
})

# --------------------------------------------------------------------------
# NON-COUNTABLE LABELS — Body parts, clothing items that shouldn't inflate counts
# Giữ trong objects[] nếu cần evidence, nhưng countable=false.
# --------------------------------------------------------------------------
NON_COUNTABLE_LABELS = frozenset({
    "HAND", "ARM", "LEG", "FACE", "HEAD", "FOOT", "FINGER",
    "TIE", "UNIFORM", "DRESS_SHIRT", "SLEEVE", "COLLAR",
})

# --------------------------------------------------------------------------
# COUNT_CANONICAL_MAP — Person-family hierarchy
# MAN/WOMAN/BOY/GIRL → PERSON for counting purposes.
# Giữ raw_label để biết model ban đầu detect gì.
# --------------------------------------------------------------------------
COUNT_CANONICAL_MAP = {
    "MAN": "PERSON",
    "WOMAN": "PERSON",
    "BOY": "PERSON",
    "GIRL": "PERSON",
    "STUDENT": "PERSON",
    "CHILD_STUDENT": "PERSON",
    "CHILD": "PERSON",
    "BABY": "PERSON",
    "TEACHER": "PERSON",
    "WORKER": "PERSON",
    "DOCTOR": "PERSON",
    "NURSE": "PERSON",
    "SOLDIER": "PERSON",
    "POLICE": "PERSON",
    "OFFICER": "PERSON",
}


PRIORITY_TAGS = [
    "person", "car", "motorcycle", "bicycle", "bus", "truck", "boat",
    "train", "airplane", "traffic light", "sign", "dog", "cat", "cow",
    "horse", "fish", "bird", "chair", "table", "sofa", "bed", "phone",
    "laptop", "bag", "backpack", "bottle", "cup", "hat", "shoe",
]


def filter_tags_for_dino(raw_tags, max_tags=None):
    """Lọc tags từ RAM++ trước khi gửi làm prompt cho GroundingDINO.

    Mục đích: Giảm prompt length → GDINO inference nhanh hơn + giảm duplicates.
    Đây là best-effort optimization, KHÔNG cần cover 100%.
    Class-agnostic NMS (Layer 2) sẽ bắt phần còn lại.

    Args:
        raw_tags: list[str] — tags gốc từ RAM++
        max_tags: int | None — giới hạn số tag object gửi sang GroundingDINO

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

    if max_tags is not None and max_tags > 0 and len(filtered) > max_tags:
        filtered = limit_prompt_tags(filtered, max_tags)

    return filtered


def limit_prompt_tags(tags, max_tags):
    """Giới hạn prompt tags nhưng ưu tiên object quan trọng cho retrieval."""
    priority_set = set(PRIORITY_TAGS)
    priority = [tag for tag in tags if tag in priority_set]
    rest = [tag for tag in tags if tag not in priority_set]

    out = []
    seen = set()
    for tag in priority + rest:
        if tag in seen:
            continue
        out.append(tag)
        seen.add(tag)
        if len(out) >= max_tags:
            break
    return out


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
    sorted_dets = sorted(detections, key=lambda d: float(d.get("score", 0.0)), reverse=True)
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


def compute_containment(box_a, box_b):
    """Tính tỷ lệ box_a nằm trong box_b (containment ratio).

    Args:
        box_a, box_b: [x1, y1, x2, y2]

    Returns:
        float: Tỷ lệ diện tích phần giao / diện tích box_a ∈ [0, 1]
    """
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    intersection = (x1 - x0) * (y1 - y0)
    area_a = max(0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    return intersection / area_a if area_a > 0 else 0.0


def canonical_count_label(label: str) -> str:
    """Trả về canonical label cho counting (WOMAN → PERSON, etc.).

    Args:
        label: str — label uppercase đã normalize

    Returns:
        str: canonical label (uppercase)
    """
    return COUNT_CANONICAL_MAP.get(label, label)


def family_aware_nms(detections, iou_threshold=0.5, containment_threshold=0.7):
    """Loại bbox trùng trong cùng person-family hierarchy.

    Khi PERSON và MAN/WOMAN overlap cao (IoU hoặc containment), giữ detection
    confidence tốt hơn, gán canonical label = PERSON, giữ raw_label gốc.

    Args:
        detections: list[dict] — {"label", "score", "box", ...}
        iou_threshold: float — IoU threshold cho family overlap
        containment_threshold: float — containment threshold (box nhỏ nằm trong box lớn)

    Returns:
        list[dict]: detections đã xử lý family hierarchy
    """
    if not detections:
        return []

    # Tách family members vs non-family
    family_labels = set(COUNT_CANONICAL_MAP.keys()) | set(COUNT_CANONICAL_MAP.values())
    family_dets = []
    non_family_dets = []

    for det in detections:
        if det.get("label", "") in family_labels:
            family_dets.append(det)
        else:
            non_family_dets.append(det)

    if len(family_dets) <= 1:
        return detections

    # Sort by score descending
    family_dets.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
    kept_family = []

    for det in family_dets:
        is_duplicate = False
        for existing in kept_family:
            iou = compute_iou(det["box"], existing["box"])
            containment = compute_containment(det["box"], existing["box"])
            reverse_containment = compute_containment(existing["box"], det["box"])
            max_containment = max(containment, reverse_containment)

            if iou > iou_threshold or max_containment > containment_threshold:
                # Overlap detected — merge: giữ existing (score cao hơn),
                # nhưng lưu raw_label nếu cần
                if "attributes" not in existing:
                    existing["attributes"] = []
                raw = det.get("raw_label") or det.get("label", "")
                if raw.lower() not in [a.lower() for a in existing["attributes"]]:
                    existing["attributes"].append(raw.lower())
                is_duplicate = True
                break

        if not is_duplicate:
            # Gán canonical label cho counting
            det_copy = dict(det)
            raw_label = det_copy.get("label", "")
            canonical = canonical_count_label(raw_label)
            if canonical != raw_label:
                det_copy["raw_label"] = raw_label
                det_copy["label"] = canonical
                if "attributes" not in det_copy:
                    det_copy["attributes"] = [raw_label.lower()]
            kept_family.append(det_copy)

    return non_family_dets + kept_family


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


def clamp_box(bbox, img_width, img_height):
    """Clamp bbox vào biên ảnh và chuẩn hóa thứ tự tọa độ."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(float(img_width), x1))
    y1 = max(0.0, min(float(img_height), y1))
    x2 = max(0.0, min(float(img_width), x2))
    y2 = max(0.0, min(float(img_height), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def normalize_and_filter(detections, img_width, img_height,
                         scene_threshold=SCENE_AREA_THRESHOLD):
    """Chuẩn hóa labels + loại bbox scene-level.

    Scene-level detection bao gồm:
    1. Bbox chiếm > scene_threshold diện tích ảnh.
    2. Label nằm trong SCENE_LABELS (bất kể kích thước bbox).

    Args:
        detections: list[dict] — {"label", "score", "box"}
        img_width, img_height: kích thước ảnh
        scene_threshold: ngưỡng area ratio tối đa

    Returns:
        tuple: (kept_objects, scene_labels)
            - kept_objects: list[dict] — objects hợp lệ, label đã normalize
            - scene_labels: list[str] — labels bị loại (chuyển vào scene_tags)
    """
    kept = []
    scene_labels = []

    for det in detections:
        box = det.get("box")
        label = det.get("label")
        if not label or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue

        try:
            clean_box = clamp_box(box, img_width, img_height)
        except (TypeError, ValueError):
            continue

        if clean_box[2] <= clean_box[0] or clean_box[3] <= clean_box[1]:
            continue

        clean_det = dict(det)
        clean_det["label"] = normalize_label(label)
        clean_det["box"] = clean_box

        area_ratio = compute_area_ratio(clean_box, img_width, img_height)

        # Check scene-level: by area OR by label name
        is_scene_by_area = area_ratio > scene_threshold
        is_scene_by_label = clean_det["label"] in SCENE_LABELS

        if is_scene_by_area or is_scene_by_label:
            scene_labels.append(clean_det["label"].lower())
            continue

        kept.append(clean_det)

    return kept, scene_labels


# ============================================================================
# LAYER 4 — Geometry Enrichment
# ============================================================================

def compute_position(bbox, img_width, img_height):
    """Tính vị trí tương đối của bbox trong ảnh dựa trên center point.

    Chia ảnh thành grid 3x3:
        top_left    | top_center    | top_right
        center_left | center        | center_right
        bottom_left | bottom_center | bottom_right

    Simplified thành left/center/right nếu chỉ cần ngang.

    Args:
        bbox: [x1, y1, x2, y2]
        img_width, img_height: kích thước ảnh

    Returns:
        str: vị trí ("left", "center", "right", "top_left", etc.)
    """
    if img_width <= 0 or img_height <= 0:
        return "center"

    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    rx = cx / img_width
    ry = cy / img_height

    # Horizontal position
    if rx < 1.0 / 3:
        h_pos = "left"
    elif rx > 2.0 / 3:
        h_pos = "right"
    else:
        h_pos = "center"

    # Vertical position
    if ry < 1.0 / 3:
        v_pos = "top"
    elif ry > 2.0 / 3:
        v_pos = "bottom"
    else:
        v_pos = "center"

    # Combine
    if v_pos == "center" and h_pos == "center":
        return "center"
    if v_pos == "center":
        return h_pos
    if h_pos == "center":
        return v_pos
    return f"{v_pos}_{h_pos}"


def size_bucket(area_ratio: float) -> str:
    """Phân loại kích thước bbox dựa trên area ratio.

    Args:
        area_ratio: float — tỷ lệ diện tích bbox / ảnh

    Returns:
        str: "small" | "medium" | "large"
    """
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.08:
        return "medium"
    return "large"


def enrich_detection(det, img_width, img_height):
    """Thêm geometry metadata cho một detection.

    Thêm các field:
        - label_lower: lowercase label
        - raw_label: label gốc trước canonical (nếu chưa có)
        - confidence: alias cho score
        - bbox_xyxy: alias cho box
        - area_ratio: tỷ lệ diện tích
        - center_xy: tọa độ tâm bbox
        - position: vị trí tương đối trong ảnh
        - size_bucket: phân loại kích thước
        - countable: có nên đếm trong object_counts không

    Args:
        det: dict — detection {"label", "score", "box", ...}
        img_width, img_height: kích thước ảnh

    Returns:
        dict: detection đã enriched (modified in-place)
    """
    box = det.get("box", [0, 0, 0, 0])
    label = det.get("label", "")
    score = det.get("score", 0.0)

    # Geometry
    area_ratio_val = compute_area_ratio(box, img_width, img_height)
    cx = round((box[0] + box[2]) / 2.0, 2)
    cy = round((box[1] + box[3]) / 2.0, 2)

    # Aliases
    det["label_lower"] = label.lower().replace("_", " ")
    if "raw_label" not in det:
        det["raw_label"] = label
    det["confidence"] = score
    det["bbox_xyxy"] = list(box)
    det["area_ratio"] = round(area_ratio_val, 4)
    det["center_xy"] = [cx, cy]
    det["position"] = compute_position(box, img_width, img_height)
    det["size_bucket"] = size_bucket(area_ratio_val)

    # Countable flag
    is_scene = label in SCENE_LABELS
    is_non_countable = label in NON_COUNTABLE_LABELS
    det["countable"] = not (is_scene or is_non_countable)

    return det


# ============================================================================
# CONVENIENCE — Full canonicalization pipeline
# ============================================================================

def canonicalize_detections(raw_detections, img_width, img_height,
                           nms_iou_threshold=0.7,
                           scene_area_threshold=SCENE_AREA_THRESHOLD,
                           family_iou_threshold=0.5,
                           family_containment_threshold=0.7):
    """Chạy toàn bộ Layer 2 + 2b + 3 + 4 trên raw detections.

    Args:
        raw_detections: list[dict] — output thô từ GroundingDINO
        img_width, img_height: kích thước ảnh
        nms_iou_threshold: IoU threshold cho NMS
        scene_area_threshold: area ratio threshold cho scene filter
        family_iou_threshold: IoU threshold cho person-family NMS
        family_containment_threshold: containment threshold cho person-family NMS

    Returns:
        tuple: (final_objects, scene_labels, quality_stats)
    """
    # Layer 2: Class-agnostic NMS
    deduped = class_agnostic_nms(raw_detections, iou_threshold=nms_iou_threshold)

    # Layer 3: Normalize + scene filter
    objects_after_scene, scene_labels = normalize_and_filter(
        deduped, img_width, img_height, scene_threshold=scene_area_threshold
    )

    # Layer 2b: Family-aware NMS (person hierarchy dedup)
    num_before_family = len(objects_after_scene)
    final_objects = family_aware_nms(
        objects_after_scene,
        iou_threshold=family_iou_threshold,
        containment_threshold=family_containment_threshold,
    )
    num_deduped_hierarchy = num_before_family - len(final_objects)

    # Layer 4: Enrich each detection with geometry metadata
    for det in final_objects:
        enrich_detection(det, img_width, img_height)

    # Quality stats
    num_countable = sum(1 for d in final_objects if d.get("countable", True))
    num_non_countable = sum(1 for d in final_objects if not d.get("countable", True))

    quality_stats = {
        "num_deduped_hierarchy": num_deduped_hierarchy,
        "num_countable_objects": num_countable,
        "num_scene_labels": len(scene_labels),
        "num_non_countable": num_non_countable,
    }

    return final_objects, scene_labels, quality_stats


# ============================================================================
# CAPTION / SEARCH FIELD BUILDERS
# ============================================================================

def build_object_counts(objects):
    """Build object_counts (legacy uppercase) và object_counts_normalized (lowercase).

    Chỉ đếm countable objects, bỏ qua scene labels và non-countable labels.

    Args:
        objects: list[dict] — enriched detections

    Returns:
        tuple: (object_counts, object_counts_normalized, object_count_items)
    """
    from collections import Counter

    countable = [obj for obj in objects if obj.get("countable", True)]
    counts_upper = Counter(obj["label"] for obj in countable)
    counts_lower = Counter(obj.get("label_lower", obj["label"].lower()) for obj in countable)

    object_counts = dict(counts_upper)
    object_counts_normalized = dict(counts_lower)
    object_count_items = [
        {"label": label, "count": count}
        for label, count in sorted(counts_lower.items(), key=lambda x: (-x[1], x[0]))
    ]

    return object_counts, object_counts_normalized, object_count_items


def build_object_text(object_counts_normalized):
    """Tạo text field cho lexical search.

    Lặp label theo count để search ưu tiên frame có nhiều object.
    Ví dụ: {"person": 2, "screen": 1} → "person person screen"

    Args:
        object_counts_normalized: dict[str, int]

    Returns:
        str: text field cho search index
    """
    parts = []
    for label, count in sorted(object_counts_normalized.items(), key=lambda x: (-x[1], x[0])):
        parts.extend([label] * count)
    return " ".join(parts)


def build_important_objects(object_counts_normalized):
    """Tạo list human-readable object descriptions cho caption.

    Ví dụ: {"person": 2, "screen": 1} → ["2 persons", "1 screen"]

    Args:
        object_counts_normalized: dict[str, int]

    Returns:
        list[str]: danh sách mô tả
    """
    items = []
    for label, count in sorted(object_counts_normalized.items(), key=lambda x: (-x[1], x[0])):
        # Simple English pluralization
        if count > 1:
            if label.endswith("s") or label.endswith("sh") or label.endswith("ch"):
                plural = label + "es"
            elif label.endswith("y") and label[-2:] not in ("ay", "ey", "oy", "uy"):
                plural = label[:-1] + "ies"
            else:
                plural = label + "s"
            items.append(f"{count} {plural}")
        else:
            items.append(f"{count} {label}")
    return items


def build_search_fields(objects, scene_labels, ram_tags):
    """Build tất cả search/caption fields từ enriched objects.

    Args:
        objects: list[dict] — enriched detections (đã có countable, label_lower)
        scene_labels: list[str] — labels bị loại ra scene
        ram_tags: list[str] — RAM++ tags gốc (lowercase)

    Returns:
        dict: chứa tất cả search/caption fields
    """
    # Object counts
    obj_counts, obj_counts_norm, obj_count_items = build_object_counts(objects)

    # Tags
    object_tags = sorted(set(
        obj.get("label_lower", obj["label"].lower())
        for obj in objects if obj.get("countable", True)
    ))
    scene_tags = sorted(set(scene_labels))
    ram_tag_list = sorted(set(t.lower() for t in ram_tags))

    # Text fields
    object_text = build_object_text(obj_counts_norm)
    scene_text = " ".join(scene_tags)
    ram_tag_text = " ".join(ram_tag_list)
    all_object_text = " ".join(filter(None, [object_text, scene_text, ram_tag_text]))

    # Important objects
    important_objects = build_important_objects(obj_counts_norm)

    return {
        "object_counts": obj_counts,
        "object_counts_normalized": obj_counts_norm,
        "object_count_items": obj_count_items,
        "object_tags": object_tags,
        "scene_tags": scene_tags,
        "ram_tags": ram_tag_list,
        "object_text": object_text,
        "scene_text": scene_text,
        "ram_tag_text": ram_tag_text,
        "all_object_text": all_object_text,
        "important_objects": important_objects,
    }
