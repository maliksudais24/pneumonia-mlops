"""
tests/test_data.py — lightweight tests that run in CI on every push.

IMPORTANT: these tests do NOT require the full dataset or a trained
model. CI runners don't have your 1GB+ dataset downloaded, and we
don't want CI to take 20 minutes — the whole point is FAST feedback.

Instead, these tests check that the core LOGIC (preprocessing, image
handling) is correct, using small generated test images.
"""

import sys
import os
import numpy as np
from PIL import Image
import io

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def create_dummy_image_bytes(size=(300, 200), mode="L"):
    """Create a small fake X-ray-like image in memory, no file needed."""
    img = Image.new(mode, size, color=128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_preprocess_image_output_shape():
    """
    Confirms our preprocessing function (from app.py) always outputs
    the exact shape the model expects: (1, 224, 224, 3).
    This is the training/serving skew check — if this shape is ever
    wrong, predictions would silently fail or error out.
    """
    from app import preprocess_image

    dummy_bytes = create_dummy_image_bytes()
    result = preprocess_image(dummy_bytes)

    assert result.shape == (1, 224, 224, 3), \
        f"Expected shape (1, 224, 224, 3), got {result.shape}"
    print("PASS: preprocess_image produces correct shape")


def test_preprocess_image_value_range():
    """
    Confirms pixel values are normalized to [0, 1], not left as [0, 255].
    If normalization is ever accidentally removed, the model would
    receive wildly out-of-range inputs and predictions would be garbage.
    """
    from app import preprocess_image

    dummy_bytes = create_dummy_image_bytes()
    result = preprocess_image(dummy_bytes)

    assert result.min() >= 0.0 and result.max() <= 1.0, \
        f"Expected values in [0,1], got min={result.min()}, max={result.max()}"
    print("PASS: preprocess_image normalizes values correctly")


def test_preprocess_handles_grayscale_input():
    """
    X-rays are often grayscale, but our model expects 3-channel RGB.
    Confirms grayscale images get converted correctly, not rejected.
    """
    from app import preprocess_image

    dummy_bytes = create_dummy_image_bytes(mode="L")  # "L" = grayscale
    result = preprocess_image(dummy_bytes)

    assert result.shape[-1] == 3, \
        f"Expected 3 color channels after conversion, got {result.shape[-1]}"
    print("PASS: grayscale images correctly converted to 3-channel")


if __name__ == "__main__":
    test_preprocess_image_output_shape()
    test_preprocess_image_value_range()
    test_preprocess_handles_grayscale_input()
    print("\nAll tests passed.")