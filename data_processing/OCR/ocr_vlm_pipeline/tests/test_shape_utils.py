from src.shape_utils import json_safe


def test_json_safe_converts_numpy_arrays_and_scalars():
    import json
    import numpy as np

    payload = {
        "bbox": np.array([1, 2, 3, 4]),
        "score": np.float32(0.5),
        "nested": {"ids": np.array([7, 8])},
    }
    safe = json_safe(payload)
    assert safe == {"bbox": [1, 2, 3, 4], "score": 0.5, "nested": {"ids": [7, 8]}}
    json.dumps(safe)
