import pytest
import numpy as np

try:
    from fenn.experimental.vision import resize_batch
    RESIZE_AVAILABLE = True
except ImportError:
    RESIZE_AVAILABLE = False


@pytest.mark.skipif(not RESIZE_AVAILABLE, reason="resize_batch not available (torchvision required)")
class TestResizeBatch:
    """Test suite for resize_batch function."""

    def test_basic_resize_channels_last(self):
        """Test basic resize with channels last format."""
        # Create a simple test image: (1, 100, 100, 3)
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        result = resize_batch(array, size=(50, 50))

        assert result.shape == (1, 50, 50, 3)
        assert result.dtype == np.uint8
        assert np.all(result >= 0) and np.all(result <= 255)

    def test_basic_resize_channels_first(self):
        """Test basic resize with channels first format."""
        # Create a test image: (1, 3, 100, 100)
        array = np.random.randint(0, 255, (1, 3, 100, 100), dtype=np.uint8)
        result = resize_batch(array, size=(50, 50))

        assert result.shape == (1, 3, 50, 50)
        assert result.dtype == np.uint8

    def test_grayscale_no_channels(self):
        """Test resize with grayscale (N, H, W)."""
        array = np.random.randint(0, 255, (1, 100, 100), dtype=np.uint8)
        result = resize_batch(array, size=(50, 50))

        assert result.shape == (1, 50, 50)
        assert result.dtype == np.uint8

    def test_float32_preservation(self):
        """Test that float32 dtype is preserved."""
        array = np.random.rand(1, 100, 100, 3).astype(np.float32)
        result = resize_batch(array, size=(50, 50))

        assert result.shape == (1, 50, 50, 3)
        assert result.dtype == np.float32
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_different_interpolation_modes(self):
        """Test different interpolation modes."""
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)

        for mode in ["nearest", "bilinear", "bicubic"]:
            result = resize_batch(array, size=(50, 50), interpolation=mode)
            assert result.shape == (1, 50, 50, 3)
            assert result.dtype == np.uint8

    def test_square_size_int(self):
        """Test resize with integer size (square output)."""
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        result = resize_batch(array, size=50)

        assert result.shape == (1, 50, 50, 3)

    def test_batch_multiple_images(self):
        """Test resize with multiple images in batch."""
        array = np.random.randint(0, 255, (5, 100, 100, 3), dtype=np.uint8)
        result = resize_batch(array, size=(50, 50))

        assert result.shape == (5, 50, 50, 3)
        assert result.dtype == np.uint8

    def test_invalid_size(self):
        """Test that invalid size raises ValueError."""
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            resize_batch(array, size=(-10, 10))

        with pytest.raises(ValueError):
            resize_batch(array, size=(10, -10))

    def test_invalid_interpolation(self):
        """Test that invalid interpolation raises ValueError."""
        array = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            resize_batch(array, size=(50, 50), interpolation="invalid_mode")

    def test_invalid_array_type(self):
        """Test that non-numpy array raises TypeError."""
        with pytest.raises(TypeError):
            resize_batch([1, 2, 3], size=(50, 50))

    def test_preserve_channel_order(self):
        """Test that channel order is preserved."""
        # Channels first
        array_cf = np.random.randint(0, 255, (1, 3, 100, 100), dtype=np.uint8)
        result_cf = resize_batch(array_cf, size=(50, 50))
        assert result_cf.shape == (1, 3, 50, 50)  # Still channels first

        # Channels last
        array_cl = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        result_cl = resize_batch(array_cl, size=(50, 50))
        assert result_cl.shape == (1, 50, 50, 3)  # Still channels last
