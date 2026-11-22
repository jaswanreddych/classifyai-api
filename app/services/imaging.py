"""Turn uploaded bytes into the array the model expects."""

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.errors import InvalidImageError

# The model was trained on 299x299 RGB images (Xception's input size).
IMAGE_SIZE = (299, 299)

# Refuse absurdly large images before decoding them (decompression bombs).
Image.MAX_IMAGE_PIXELS = 25_000_000


def preprocess(data: bytes) -> np.ndarray:
    """Decode an image and return a float32 array of shape (299, 299, 3).

    Pixel values stay in 0-255. The model rescales them to 0-1 itself.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            rgb = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise InvalidImageError("The file is not a readable image.") from exc
    return np.asarray(rgb, dtype=np.float32)
