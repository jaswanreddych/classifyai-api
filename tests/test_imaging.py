import numpy as np
import pytest

from app.errors import InvalidImageError
from app.services.imaging import IMAGE_SIZE, preprocess
from tests.conftest import make_image


def test_output_shape_dtype_and_range():
    array = preprocess(make_image(size=(120, 80)))
    assert array.shape == (*IMAGE_SIZE, 3)
    assert array.dtype == np.float32
    assert array.min() >= 0 and array.max() <= 255


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK"])
def test_other_color_modes_become_three_channel_rgb(mode):
    fmt = "TIFF" if mode == "CMYK" else "PNG"
    assert preprocess(make_image(fmt=fmt, mode=mode)).shape == (*IMAGE_SIZE, 3)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_supported_formats(fmt):
    assert preprocess(make_image(fmt=fmt)).shape == (*IMAGE_SIZE, 3)


def test_garbage_bytes_raise_invalid_image():
    with pytest.raises(InvalidImageError):
        preprocess(b"this is not an image")


def test_truncated_image_raises_invalid_image():
    with pytest.raises(InvalidImageError):
        preprocess(make_image()[:40])
