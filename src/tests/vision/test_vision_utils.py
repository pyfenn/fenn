import pytest
import numpy as np

from fenn.vision.vision_utils import detect_format


class TestDetectFormat:
    """Test suite for detect_format function."""

    def test_3d_grayscale(self):
        """Test 3D grayscale array (N, H, W)."""
        array = np.zeros((10, 224, 224), dtype=np.uint8)
        result = detect_format(array)
        
        assert result["is_grayscale"] is True
        assert result["channel_location"] is None

    def test_4d_channels_last_color(self):
        """Test 4D color array with channels last (N, H, W, C)."""
        array = np.zeros((10, 224, 224, 3), dtype=np.uint8)
        result = detect_format(array)
        
        assert result["is_grayscale"] is False
        assert result["channel_location"] == "last"

    def test_4d_channels_first_color(self):
        """Test 4D color array with channels first (N, C, H, W)."""
        array = np.zeros((10, 3, 224, 224), dtype=np.uint8)
        result = detect_format(array)
        
        assert result["is_grayscale"] is False
        assert result["channel_location"] == "first"

    def test_4d_grayscale_channels_last(self):
        """Test 4D grayscale array with channels last (N, H, W, 1)."""
        array = np.zeros((10, 224, 224, 1), dtype=np.uint8)
        result = detect_format(array)
        
        assert result["is_grayscale"] is True
        assert result["channel_location"] == "last"

    def test_4d_grayscale_channels_first(self):
        """Test 4D grayscale array with channels first (N, 1, H, W)."""
        array = np.zeros((10, 1, 224, 224), dtype=np.uint8)
        result = detect_format(array)
        
        assert result["is_grayscale"] is True
        assert result["channel_location"] == "first"

    def test_4d_rgba_channels_last(self):
        """Test 4D RGBA array with channels last (N, H, W, 4)."""
        array = np.zeros((10, 224, 224, 4), dtype=np.uint8)
        result = detect_format(array)
        
        assert result["is_grayscale"] is False
        assert result["channel_location"] == "last"

    def test_2d_array_raises_error(self):
        """Test that 2D array raises ValueError."""
        array = np.zeros((224, 224), dtype=np.uint8)
        
        with pytest.raises(ValueError) as exc_info:
            detect_format(array)
        
        assert "batch dimension" in str(exc_info.value).lower()

    def test_non_numpy_array_raises_error(self):
        """Test that non-numpy array raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            detect_format([1, 2, 3])
        
        assert "numpy.ndarray" in str(exc_info.value)

    def test_5d_array_raises_error(self):
        """Test that 5D array raises ValueError."""
        array = np.zeros((10, 3, 224, 224, 1), dtype=np.uint8)
        
        with pytest.raises(ValueError) as exc_info:
            detect_format(array)
        
        assert "Unsupported array dimension" in str(exc_info.value)

    def test_4d_invalid_format_raises_error(self):
        """Test that invalid 4D format raises ValueError."""
        # Array where neither dimension looks like channels (both > 4)
        array = np.zeros((10, 100, 224, 224), dtype=np.uint8)
        
        with pytest.raises(ValueError) as exc_info:
            detect_format(array)
        
        assert "Unable to detect format" in str(exc_info.value)

