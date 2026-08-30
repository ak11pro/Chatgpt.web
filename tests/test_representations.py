import pytest

np = pytest.importorskip("numpy")

from hotc_tracker.representations import pca_rgb, robust_band_rgb


def test_rgb_representations_return_uint8_three_channel_image() -> None:
    cube = np.arange(4 * 5 * 16, dtype=np.float32).reshape(4, 5, 16)
    for image in (robust_band_rgb(cube), pca_rgb(cube)):
        assert image.shape == (4, 5, 3)
        assert image.dtype == np.uint8
