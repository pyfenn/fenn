from unittest.mock import patch

import numpy as np
import pytest

try:
    from fenn.experimental.vision import resize_batch

    RESIZE_AVAILABLE = True
except ImportError:
    RESIZE_AVAILABLE = False


@pytest.mark.skipif(
    not RESIZE_AVAILABLE, reason="resize_batch not available (torchvision required)"
)
class TestResizeBatch:
    """Test suite for resize_batch function."""

    # ------------------------------------------------------------------
    # Basic shape / format preservation
    # ------------------------------------------------------------------

    def test_basic_resize_channels_last(self):
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        result = resize_batch(array, size=(50, 50))
        assert result.shape == (1, 50, 50, 3)
        assert result.dtype == np.uint8
        assert np.all(result >= 0) and np.all(result <= 255)

    def test_basic_resize_channels_first(self):
        array = np.random.randint(0, 255, (1, 3, 100, 100), dtype=np.uint8)
        result = resize_batch(array, size=(50, 50))
        assert result.shape == (1, 3, 50, 50)
        assert result.dtype == np.uint8

    def test_grayscale_no_channels(self):
        array = np.random.randint(0, 255, (1, 100, 100), dtype=np.uint8)
        result = resize_batch(array, size=(50, 50))
        assert result.shape == (1, 50, 50)
        assert result.dtype == np.uint8

    def test_preserve_channel_order(self):
        array_cf = np.random.randint(0, 255, (1, 3, 100, 100), dtype=np.uint8)
        result_cf = resize_batch(array_cf, size=(50, 50))
        assert result_cf.shape == (1, 3, 50, 50)

        array_cl = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        result_cl = resize_batch(array_cl, size=(50, 50))
        assert result_cl.shape == (1, 50, 50, 3)

    def test_batch_multiple_images(self):
        array = np.random.randint(0, 255, (5, 100, 100, 3), dtype=np.uint8)
        result = resize_batch(array, size=(50, 50))
        assert result.shape == (5, 50, 50, 3)
        assert result.dtype == np.uint8

    # ------------------------------------------------------------------
    # Size argument variants
    # ------------------------------------------------------------------

    def test_square_size_int(self):
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        result = resize_batch(array, size=50)
        assert result.shape == (1, 50, 50, 3)

    def test_non_square_tuple_size(self):
        """Rectangular target size should produce non-square output."""
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        result = resize_batch(array, size=(40, 80))
        assert result.shape == (1, 40, 80, 3)

    def test_upsampling(self):
        """Upsampling (target > source) should disable antialias and still work."""
        array = np.random.randint(0, 255, (1, 50, 50, 3), dtype=np.uint8)
        result = resize_batch(array, size=(100, 100))
        assert result.shape == (1, 100, 100, 3)
        assert result.dtype == np.uint8

    def test_same_size_noop(self):
        """Resizing to same dimensions should return identically shaped array."""
        array = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
        result = resize_batch(array, size=(64, 64))
        assert result.shape == array.shape

    # ------------------------------------------------------------------
    # dtype handling
    # ------------------------------------------------------------------

    def test_float32_preservation(self):
        array = np.random.rand(1, 100, 100, 3).astype(np.float32)
        result = resize_batch(array, size=(50, 50))
        assert result.shape == (1, 50, 50, 3)
        assert result.dtype == np.float32
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_float64_preservation(self):
        """float64 input should be preserved and values clipped to [0, 1]."""
        array = np.random.rand(1, 100, 100, 3).astype(np.float64)
        result = resize_batch(array, size=(50, 50))
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_uint16_normalization(self):
        """uint16 is neither uint8 nor float; should normalize through float and back."""
        array = np.random.randint(0, 65535, (1, 100, 100, 3), dtype=np.uint16)
        result = resize_batch(array, size=(50, 50))
        assert result.shape == (1, 50, 50, 3)
        assert result.dtype == np.uint16
        assert np.all(result >= 0) and np.all(result <= 65535)

    def test_int32_normalization(self):
        """int32 should go through float normalization path."""
        array = np.random.randint(0, 1000, (1, 100, 100, 3), dtype=np.int32)
        result = resize_batch(array, size=(50, 50))
        assert result.dtype == np.int32

    # ------------------------------------------------------------------
    # Interpolation modes
    # ------------------------------------------------------------------

    def test_interpolation_nearest(self):
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        result = resize_batch(array, size=(50, 50), interpolation="nearest")
        assert result.shape == (1, 50, 50, 3)

    def test_interpolation_bilinear(self):
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        result = resize_batch(array, size=(50, 50), interpolation="bilinear")
        assert result.shape == (1, 50, 50, 3)

    def test_interpolation_bicubic(self):
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        result = resize_batch(array, size=(50, 50), interpolation="bicubic")
        assert result.shape == (1, 50, 50, 3)

    def test_interpolation_nearest_exact(self):
        """nearest_exact is a valid mode that the original tests omitted."""
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        result = resize_batch(array, size=(50, 50), interpolation="nearest_exact")
        assert result.shape == (1, 50, 50, 3)

    def test_different_interpolation_modes(self):
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        for mode in ["nearest", "bilinear", "bicubic"]:
            result = resize_batch(array, size=(50, 50), interpolation=mode)
            assert result.shape == (1, 50, 50, 3)
            assert result.dtype == np.uint8

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    def test_invalid_size_negative_height(self):
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            resize_batch(array, size=(-10, 10))

    def test_invalid_size_negative_width(self):
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            resize_batch(array, size=(10, -10))

    def test_invalid_size_zero(self):
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            resize_batch(array, size=(0, 50))

    def test_invalid_size_wrong_type(self):
        """A non-int, non-2-tuple size should raise ValueError."""
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            resize_batch(array, size=(10, 10, 10))  # 3-tuple

    def test_invalid_size_string(self):
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        with pytest.raises((ValueError, TypeError)):
            resize_batch(array, size="large")

    def test_invalid_interpolation(self):
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            resize_batch(array, size=(50, 50), interpolation="invalid_mode")

    def test_invalid_array_type_list(self):
        with pytest.raises(TypeError):
            resize_batch([1, 2, 3], size=(50, 50))

    def test_invalid_array_type_none(self):
        with pytest.raises(TypeError):
            resize_batch(None, size=(50, 50))

    def test_torchvision_unavailable(self):
        """When torchvision is not installed, ImportError should be raised."""
        import fenn.experimental.vision.resize as resize_module  # adjust to real module path

        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        with patch.object(resize_module, "TORCHVISION_AVAILABLE", False):
            with pytest.raises(ImportError, match="torchvision"):
                resize_batch(array, size=(50, 50))

    # ------------------------------------------------------------------
    # Antialias behaviour (white-box)
    # ------------------------------------------------------------------

    def test_antialias_disabled_for_nearest_downsampling(self):
        """nearest interpolation should never trigger antialias even when downsampling."""
        import torchvision.transforms.functional as F

        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        calls = []
        original_resize = F.resize

        def mock_resize(tensor, size, interpolation, antialias):
            calls.append(antialias)
            return original_resize(
                tensor, size, interpolation=interpolation, antialias=antialias
            )

        with patch.object(F, "resize", side_effect=mock_resize):
            resize_batch(array, size=(50, 50), interpolation="nearest")

        assert calls and calls[0] is False

    def test_antialias_enabled_for_bilinear_downsampling(self):
        """bilinear downsampling should enable antialias."""
        import torchvision.transforms.functional as F

        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        calls = []
        original_resize = F.resize

        def mock_resize(tensor, size, interpolation, antialias):
            calls.append(antialias)
            return original_resize(
                tensor, size, interpolation=interpolation, antialias=antialias
            )

        with patch.object(F, "resize", side_effect=mock_resize):
            resize_batch(array, size=(50, 50), interpolation="bilinear")

        assert calls and calls[0] is True
