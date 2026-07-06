# -*- coding: utf-8 -*-
"""Tag filtering and prompt building for RAM++ -> LocateAnything."""

from __future__ import annotations

from ram_locate_common import unique_preserve_order


SYNONYM_MAP = {
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
    "goose": "duck",
    "van": "car",
    "suv": "car",
    "sedan": "car",
    "automobile": "car",
    "motorbike": "motorcycle",
    "scooter": "motorcycle",
    "bike": "bicycle",
    "cell phone": "phone",
    "mobile phone": "phone",
    "smartphone": "phone",
    "baseball cap": "hat",
    "cowboy hat": "hat",
    "sun hat": "hat",
    "beanie": "hat",
    "cap": "hat",
    "sneaker": "shoe",
    "boot": "shoe",
    "sandal": "shoe",
    "mitten": "glove",
    "couch": "sofa",
    "stool": "chair",
}


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
        "bench", "couch", "stool",
    ],
    "clothing": [
        "shirt", "dress", "hat", "shoe", "glove", "jacket", "coat",
        "pants", "skirt", "suit", "uniform",
    ],
    "plant": ["tree", "flower", "grass", "bush", "palm"],
}


NON_OBJECT_TAGS = frozenset({
    "area", "surround", "lush", "grassy", "open", "outdoor", "indoor",
    "rural", "urban", "aerial", "panoramic", "close up", "background",
    "foreground", "landscape", "scenery", "horizon", "scene", "view",
    "city view", "city skyline", "skyline", "night view", "night", "day",
    "home", "room", "living room", "kitchen", "street", "road", "market",
    "shop", "store", "restaurant", "hospital", "clinic", "office",
    "catch", "push", "stand", "standing", "stare", "swim", "graze",
    "wash", "scrub", "wear", "wearing", "walk", "walking", "run",
    "running", "sit", "sitting", "play", "playing", "drive", "driving",
    "fly", "ride", "riding", "eat", "drink", "cook", "read", "write",
    "talk", "sleep", "dance", "sing", "jump", "climb", "throw", "hold",
    "holding", "carry", "pull", "load",
    "red", "blue", "white", "green", "black", "yellow", "orange",
    "purple", "pink", "brown", "gray", "grey", "golden", "silver",
    "video", "news", "interview", "image", "photo", "film", "show",
    "broadcast", "channel", "program", "television",
    "large", "small", "big", "tall", "short", "long", "wide", "narrow",
    "old", "new", "young", "beautiful", "dirty", "clean", "wet", "dry",
    "operate", "operation", "surgery", "perform", "job", "procedure",
    "treatment", "therapy", "diagnosis",
})


PRIORITY_TAGS = [
    "person", "car", "motorcycle", "bicycle", "bus", "truck", "boat",
    "train", "airplane", "traffic light", "sign", "dog", "cat", "cow",
    "horse", "fish", "bird", "chair", "table", "sofa", "bed", "phone",
    "laptop", "bag", "backpack", "bottle", "cup", "hat", "shoe",
]


def normalize_tag(tag: str) -> str:
    clean = tag.lower().replace("_", " ").strip()
    clean = " ".join(clean.split())
    return SYNONYM_MAP.get(clean, clean)


def filter_tags_for_locate(raw_tags: list[str], max_tags: int | None = 20) -> list[str]:
    tags = unique_preserve_order(normalize_tag(t) for t in raw_tags)
    tags = [t for t in tags if t and t not in NON_OBJECT_TAGS]

    tag_set = set(tags)
    filtered: list[str] = []
    for tag in tags:
        hyponyms = HYPERNYM_CHAINS.get(tag)
        if hyponyms and any(h in tag_set for h in hyponyms):
            continue
        filtered.append(tag)

    filtered = unique_preserve_order(filtered)
    if max_tags is not None and max_tags > 0 and len(filtered) > max_tags:
        filtered = limit_prompt_tags(filtered, max_tags)
    return filtered


def limit_prompt_tags(tags: list[str], max_tags: int) -> list[str]:
    priority_set = set(PRIORITY_TAGS)
    priority = [t for t in tags if t in priority_set]
    rest = [t for t in tags if t not in priority_set]
    return unique_preserve_order(priority + rest)[:max_tags]


def build_locate_prompt(tag_list: list[str]) -> str:
    tags = [normalize_tag(t) for t in tag_list if str(t).strip()]
    tags = unique_preserve_order(tags)
    if not tags:
        return ""
    categories = "</c>".join(tags)
    return f"Locate all the instances that matches the following description: {categories}."

