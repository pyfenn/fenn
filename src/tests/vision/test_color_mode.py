import pytest
import numpy as np

from fenn.experimental.vision import ensure_color_mode


class TestEnsureColorMode:
    """Test suite for ensure_color_mode function."""

    def test_gray_to_rgb_channels_last(self):
        """Test GRAY → RGB conversion with channels last."""
        gray = np.random.randint(0, 255, (10, 224, 224), dtype=np.uint8)
        rgb = ensure_color_mode(gray, mode="RGB")
        
        assert rgb.shape == (10, 224, 224, 3)
        assert rgb.dtype == gray.dtype
        # All channels should be identical
        np.testing.assert_array_equal(rgb[:, :, :, 0], gray)
        np.testing.assert_array_equal(rgb[:, :, :, 1], gray)
        np.testing.assert_array_equal(rgb[:, :, :, 2], gray)

    def test_gray_to_rgb_channels_first(self):
        """Test GRAY → RGB conversion with channels first."""
        gray = np.random.randint(0, 255, (10, 1, 224, 224), dtype=np.uint8)
        rgb = ensure_color_mode(gray, mode="RGB")
        
        assert rgb.shape == (10, 3, 224, 224)
        assert rgb.dtype == gray.dtype
        # All channels should be identical
        np.testing.assert_array_equal(rgb[:, 0, :, :], gray[:, 0, :, :])
        np.testing.assert_array_equal(rgb[:, 1, :, :], gray[:, 0, :, :])
        np.testing.assert_array_equal(rgb[:, 2, :, :], gray[:, 0, :, :])

    def test_gray_to_rgba_channels_last(self):
        """Test GRAY → RGBA conversion with channels last."""
        gray = np.random.randint(0, 255, (10, 224, 224), dtype=np.uint8)
        rgba = ensure_color_mode(gray, mode="RGBA")
        
        assert rgba.shape == (10, 224, 224, 4)
        assert rgba.dtype == gray.dtype
        # RGB channels should be identical
        np.testing.assert_array_equal(rgba[:, :, :, 0], gray)
        np.testing.assert_array_equal(rgba[:, :, :, 1], gray)
        np.testing.assert_array_equal(rgba[:, :, :, 2], gray)
        # Alpha should be 255 (full opacity for uint8)
        assert np.all(rgba[:, :, :, 3] == 255)

    def test_rgb_to_gray_channels_last(self):
        """Test RGB → GRAY conversion with channels last."""
        rgb = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)
        gray = ensure_color_mode(rgb, mode="GRAY")
        
        assert gray.shape == (10, 224, 224)
        assert gray.dtype == rgb.dtype
        # Grayscale should be weighted average
        expected = (rgb[:, :, :, 0] * 0.299 + rgb[:, :, :, 1] * 0.587 + rgb[:, :, :, 2] * 0.114).astype(np.uint8)
        np.testing.assert_array_almost_equal(gray, expected, decimal=0)

    def test_rgb_to_gray_channels_first(self):
        """Test RGB → GRAY conversion with channels first."""
        rgb = np.random.randint(0, 255, (10, 3, 224, 224), dtype=np.uint8)
        gray = ensure_color_mode(rgb, mode="GRAY")
        
        assert gray.shape == (10, 224, 224)
        assert gray.dtype == rgb.dtype

    def test_rgb_to_rgba_channels_last(self):
        """Test RGB → RGBA conversion with channels last."""
        rgb = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)
        rgba = ensure_color_mode(rgb, mode="RGBA")
        
        assert rgba.shape == (10, 224, 224, 4)
        assert rgba.dtype == rgb.dtype
        # RGB channels should be preserved
        np.testing.assert_array_equal(rgba[:, :, :, :3], rgb)
        # Alpha should be 255
        assert np.all(rgba[:, :, :, 3] == 255)

    def test_rgba_to_rgb_channels_last(self):
        """Test RGBA → RGB conversion with channels last."""
        rgba = np.random.randint(0, 255, (10, 224, 224, 4), dtype=np.uint8)
        rgb = ensure_color_mode(rgba, mode="RGB")
        
        assert rgb.shape == (10, 224, 224, 3)
        assert rgb.dtype == rgba.dtype
        # RGB channels should be preserved
        np.testing.assert_array_equal(rgb, rgba[:, :, :, :3])

    def test_rgba_to_gray_channels_last(self):
        """Test RGBA → GRAY conversion with channels last."""
        rgba = np.random.randint(0, 255, (10, 224, 224, 4), dtype=np.uint8)
        gray = ensure_color_mode(rgba, mode="GRAY")
        
        assert gray.shape == (10, 224, 224)
        assert gray.dtype == rgba.dtype
        # Should match RGB → GRAY conversion
        rgb = rgba[:, :, :, :3]
        expected = (rgb[:, :, :, 0] * 0.299 + rgb[:, :, :, 1] * 0.587 + rgb[:, :, :, 2] * 0.114).astype(np.uint8)
        np.testing.assert_array_almost_equal(gray, expected, decimal=0)

    def test_no_op_rgb(self):
        """Test that RGB → RGB returns a copy."""
        rgb = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)
        result = ensure_color_mode(rgb, mode="RGB")
        
        assert result.shape == rgb.shape
        assert result.dtype == rgb.dtype
        np.testing.assert_array_equal(result, rgb)
        # Should be a copy, not the same array
        assert result is not rgb

    def test_no_op_rgba(self):
        """Test that RGBA → RGBA returns a copy."""
        rgba = np.random.randint(0, 255, (10, 224, 224, 4), dtype=np.uint8)
        result = ensure_color_mode(rgba, mode="RGBA")
        
        assert result.shape == rgba.shape
        assert result.dtype == rgba.dtype
        np.testing.assert_array_equal(result, rgba)
        assert result is not rgba

    def test_no_op_gray(self):
        """Test that GRAY → GRAY returns a copy."""
        gray = np.random.randint(0, 255, (10, 224, 224), dtype=np.uint8)
        result = ensure_color_mode(gray, mode="GRAY")
        
        assert result.shape == gray.shape
        assert result.dtype == gray.dtype
        np.testing.assert_array_equal(result, gray)
        assert result is not gray

    def test_mode_alias_l_to_gray(self):
        """Test that 'L' mode is normalized to 'GRAY'."""
        gray = np.random.randint(0, 255, (10, 224, 224), dtype=np.uint8)
        result_l = ensure_color_mode(gray, mode="L")
        result_gray = ensure_color_mode(gray, mode="GRAY")
        
        np.testing.assert_array_equal(result_l, result_gray)

    def test_float_dtype(self):
        """Test conversion with float dtype."""
        rgb = np.random.rand(10, 224, 224, 3).astype(np.float32)
        rgba = ensure_color_mode(rgb, mode="RGBA")
        
        assert rgba.shape == (10, 224, 224, 4)
        assert rgba.dtype == rgb.dtype
        # Alpha should be 1.0 for float
        assert np.allclose(rgba[:, :, :, 3], 1.0)

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        rgb = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError) as exc_info:
            ensure_color_mode(rgb, mode="INVALID")
        
        assert "Unsupported color mode" in str(exc_info.value)

    def test_non_numpy_array_raises_error(self):
        """Test that non-numpy array raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ensure_color_mode([1, 2, 3], mode="RGB")
        
        assert "numpy.ndarray" in str(exc_info.value)

    def test_preserves_dtype(self):
        """Test that dtype is preserved through conversions."""
        gray = np.random.randint(0, 255, (10, 224, 224), dtype=np.uint16)
        rgb = ensure_color_mode(gray, mode="RGB")
        
        assert rgb.dtype == gray.dtype

    def test_gray_to_rgba_to_gray_round_trip(self):
        """Test property-based round-trip: GRAY → RGBA → GRAY preserves values."""
        # Create grayscale image
        gray_original = np.random.randint(0, 255, (10, 224, 224), dtype=np.uint8)
        
        # Convert to RGBA
        rgba = ensure_color_mode(gray_original, mode="RGBA")
        assert rgba.shape == (10, 224, 224, 4)
        
        # Convert back to GRAY
        gray_restored = ensure_color_mode(rgba, mode="GRAY")
        assert gray_restored.shape == gray_original.shape
        
        # Values should be preserved (within rounding tolerance for grayscale conversion)
        np.testing.assert_array_almost_equal(gray_restored, gray_original, decimal=0)

    def test_dtype_int16(self):
        """Test conversion with int16 dtype."""
        gray = np.random.randint(0, 32767, (10, 224, 224), dtype=np.int16)
        rgb = ensure_color_mode(gray, mode="RGB")
        
        assert rgb.shape == (10, 224, 224, 3)
        assert rgb.dtype == gray.dtype
        # All channels should be identical
        np.testing.assert_array_equal(rgb[:, :, :, 0], gray)
        
        # Test RGBA conversion
        rgba = ensure_color_mode(rgb, mode="RGBA")
        assert rgba.shape == (10, 224, 224, 4)
        assert rgba.dtype == rgb.dtype
        # Alpha should be max value for int16
        assert np.all(rgba[:, :, :, 3] == np.iinfo(np.int16).max)
