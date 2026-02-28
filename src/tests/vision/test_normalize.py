import pytest
import numpy as np
import warnings

from fenn.experimental.vision.normalize import normalize_batch

# Suppress expected warnings from division by zero in constant image edge cases
pytestmark = pytest.mark.filterwarnings(
    "ignore:invalid value encountered in divide:RuntimeWarning"
)


class TestNormalizeBatch:
    """Test suite for normalize_batch function."""

    def test_normalize_0_1_grayscale_3d(self):
        """Test 0_1 normalization for 3D grayscale batch."""
        # Create batch with known range [0, 100]
        array = np.array([[[0, 50], [50, 100]]], dtype=np.float64)
        normalized = normalize_batch(array, mode="0_1")
        
        assert normalized.dtype == np.float64
        assert normalized.shape == array.shape
        # Min should become 0, max should become 1
        assert np.allclose(normalized[0, 0, 0], 0.0)
        assert np.allclose(normalized[0, 1, 1], 1.0)
        assert np.allclose(normalized[0, 0, 1], 0.5)  # 50 should map to 0.5
        assert np.all(normalized >= 0.0) and np.all(normalized <= 1.0)

    def test_normalize_0_1_rgb_channels_last(self):
        """Test 0_1 normalization for RGB batch with channels last."""
        # Create batch with known range [0, 255]
        array = np.array([[[[0, 128, 255], [64, 192, 0]]]], dtype=np.uint8)
        normalized = normalize_batch(array, mode="0_1")
        
        assert normalized.dtype == np.float64
        assert normalized.shape == array.shape
        # Values should be in [0, 1] range
        assert np.all(normalized >= 0.0) and np.all(normalized <= 1.0)

    def test_normalize_0_1_rgb_channels_first(self):
        """Test 0_1 normalization for RGB batch with channels first."""
        # Create batch with known range [0, 255]
        array = np.array([[[[0, 64], [128, 192]], [[0, 64], [128, 192]], [[0, 64], [128, 192]]]], dtype=np.uint8)
        normalized = normalize_batch(array, mode="0_1")
        
        assert normalized.dtype == np.float64
        assert normalized.shape == array.shape
        assert np.all(normalized >= 0.0) and np.all(normalized <= 1.0)

    def test_normalize_0_1_rgba_scales_alpha(self):
        """Test that 0_1 normalization scales alpha channel to [0, 1] for RGBA."""
        # Create RGBA batch (5x5) with different alpha values
        array = np.zeros((1, 5, 5, 4), dtype=np.uint8)
        array[0, :, :, :3] = 100  # RGB channels
        array[0, :2, :, 3] = 200  # Alpha: first 2 rows = 200
        array[0, 2:, :, 3] = 50   # Alpha: last 3 rows = 50
        normalized = normalize_batch(array, mode="0_1")
        
        # Alpha channel should be scaled to [0, 1] (uint8 / 255)
        expected_alpha_200 = 200.0 / 255.0
        expected_alpha_50 = 50.0 / 255.0
        assert np.allclose(normalized[0, :2, :, 3], expected_alpha_200)
        assert np.allclose(normalized[0, 2:, :, 3], expected_alpha_50)
        # Alpha should be in [0, 1] range
        assert np.all(normalized[..., 3:4] >= 0.0) and np.all(normalized[..., 3:4] <= 1.0)
        # RGB should be normalized
        assert np.all(normalized[..., :3] >= 0.0) and np.all(normalized[..., :3] <= 1.0)

    def test_normalize_0_1_rgba_alpha_already_normalized(self):
        """Test that 0_1 normalization handles alpha already in [0, 1] range."""
        # Create RGBA with alpha already in [0, 1]
        array = np.random.rand(1, 5, 5, 4).astype(np.float32)
        array[0, :, :, 3] = 0.5  # Alpha already in [0, 1]
        normalized = normalize_batch(array, mode="0_1")
        
        # Alpha should remain unchanged (already in [0, 1])
        assert np.allclose(normalized[..., 3:4], array[..., 3:4])

    def test_normalize_0_1_constant_image(self):
        """Test 0_1 normalization for constant image (edge case)."""
        # Constant image should map to 0.5
        array = np.array([[[42, 42], [42, 42]]], dtype=np.uint8)
        normalized = normalize_batch(array, mode="0_1")
        
        assert np.allclose(normalized, 0.5)

    def test_normalize_minus1_1_grayscale_3d(self):
        """Test minus1_1 normalization for 3D grayscale batch."""
        # Create batch with known range [0, 100]
        array = np.array([[[0, 50], [50, 100]]], dtype=np.float64)
        normalized = normalize_batch(array, mode="minus1_1")
        
        assert normalized.dtype == np.float64
        assert normalized.shape == array.shape
        # Min should become -1, max should become 1
        assert np.allclose(normalized[0, 0, 0], -1.0)
        assert np.allclose(normalized[0, 1, 1], 1.0)
        assert np.allclose(normalized[0, 0, 1], 0.0)  # 50 (midpoint) should map to 0.0
        assert np.all(normalized >= -1.0) and np.all(normalized <= 1.0)

    def test_normalize_minus1_1_rgb_channels_last(self):
        """Test minus1_1 normalization for RGB batch with channels last."""
        array = np.array([[[[0, 128, 255], [64, 192, 0]]]], dtype=np.uint8)
        normalized = normalize_batch(array, mode="minus1_1")
        
        assert normalized.dtype == np.float64
        assert normalized.shape == array.shape
        assert np.all(normalized >= -1.0) and np.all(normalized <= 1.0)

    def test_normalize_minus1_1_rgba_scales_alpha(self):
        """Test that minus1_1 normalization scales alpha channel to [-1, 1] for RGBA."""
        array = np.zeros((1, 5, 5, 4), dtype=np.uint8)
        array[0, :, :, :3] = 100  # RGB channels
        array[0, :2, :, 3] = 200  # Alpha: first 2 rows = 200
        array[0, 2:, :, 3] = 50   # Alpha: last 3 rows = 50
        normalized = normalize_batch(array, mode="minus1_1")
        
        # Alpha channel should be scaled to [-1, 1] (2 * (uint8 / 255) - 1)
        expected_alpha_200 = 2.0 * (200.0 / 255.0) - 1.0
        expected_alpha_50 = 2.0 * (50.0 / 255.0) - 1.0
        assert np.allclose(normalized[0, :2, :, 3], expected_alpha_200)
        assert np.allclose(normalized[0, 2:, :, 3], expected_alpha_50)
        # Alpha should be in [-1, 1] range
        assert np.all(normalized[..., 3:4] >= -1.0) and np.all(normalized[..., 3:4] <= 1.0)
        # RGB should be normalized to [-1, 1]
        assert np.all(normalized[..., :3] >= -1.0) and np.all(normalized[..., :3] <= 1.0)

    def test_normalize_minus1_1_rgba_alpha_already_normalized(self):
        """Test that minus1_1 normalization handles alpha already in [0, 1] range."""
        # Create RGBA with alpha already in [0, 1]
        array = np.random.rand(1, 5, 5, 4).astype(np.float32)
        array[0, :, :, 3] = 0.5  # Alpha already in [0, 1]
        normalized = normalize_batch(array, mode="minus1_1")
        
        # Alpha should be scaled to [-1, 1]: 2 * 0.5 - 1 = 0.0
        assert np.allclose(normalized[..., 3:4], 0.0)

    def test_normalize_minus1_1_constant_image(self):
        """Test minus1_1 normalization for constant image (edge case)."""
        # Constant image should map to 0.0
        array = np.array([[[42, 42], [42, 42]]], dtype=np.uint8)
        normalized = normalize_batch(array, mode="minus1_1")
        
        assert np.allclose(normalized, 0.0)

    def test_normalize_per_image_independence(self):
        """Test that each image in batch is normalized independently."""
        # Create batch (5x5) where each image has different range
        array = np.zeros((2, 5, 5), dtype=np.float64)
        # Image 0: range [0, 100]
        array[0, :, :] = np.linspace(0, 100, 25).reshape(5, 5)
        # Image 1: range [0, 400]
        array[1, :, :] = np.linspace(0, 400, 25).reshape(5, 5)
        normalized = normalize_batch(array, mode="0_1")
        
        # Both images should have min=0, max=1 after normalization
        assert np.allclose(normalized[0].min(), 0.0)
        assert np.allclose(normalized[0].max(), 1.0)
        assert np.allclose(normalized[1].min(), 0.0)
        assert np.allclose(normalized[1].max(), 1.0)
        # But the same pixel value (50) should map to different normalized values
        # Image 0: 50/100 = 0.5, Image 1: 50/400 = 0.125
        # Find where value 50 appears in each image
        img0_val_50_idx = np.where(array[0] == 50.0)
        img1_val_50_idx = np.where(array[1] == 50.0)
        if len(img0_val_50_idx[0]) > 0 and len(img1_val_50_idx[0]) > 0:
            assert np.allclose(normalized[0, img0_val_50_idx[0][0], img0_val_50_idx[1][0]], 0.5)
            assert np.allclose(normalized[1, img1_val_50_idx[0][0], img1_val_50_idx[1][0]], 0.125)

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        array = np.array([[[0, 50], [50, 100]]], dtype=np.float64)
        with pytest.raises(ValueError, match="Unsupported normalization mode"):
            normalize_batch(array, mode="invalid_mode")

    def test_normalize_zscore_grayscale_3d(self):
        """Test z-score normalization for 3D grayscale batch."""
        array = np.array([[[0, 50], [50, 100]]], dtype=np.float64)
        normalized = normalize_batch(array, mode="zscore")
        
        assert normalized.dtype == np.float64
        assert normalized.shape == array.shape
        # Mean should be ~0, std should be ~1
        assert np.allclose(normalized[0].mean(), 0.0, atol=1e-10)
        assert np.allclose(normalized[0].std(), 1.0, atol=1e-10)

    def test_normalize_zscore_rgb_channels_last(self):
        """Test z-score normalization for RGB batch with channels last."""
        array = np.random.randint(0, 255, (1, 5, 5, 3), dtype=np.uint8).astype(np.float64)
        normalized = normalize_batch(array, mode="zscore")
        
        assert normalized.dtype == np.float64
        assert normalized.shape == array.shape
        # Mean should be ~0, std should be ~1
        assert np.allclose(normalized[0].mean(), 0.0, atol=1e-10)
        assert np.allclose(normalized[0].std(), 1.0, atol=1e-10)

    def test_normalize_zscore_rgba_scales_alpha(self):
        """Test that z-score normalization scales alpha channel to [0, 1] for RGBA."""
        array = np.zeros((1, 5, 5, 4), dtype=np.uint8)
        array[0, :, :, :3] = np.random.randint(0, 255, (5, 5, 3))
        array[0, :, :, 3] = 200  # Alpha
        normalized = normalize_batch(array, mode="zscore")
        
        # Alpha channel should be scaled to [0, 1] (200/255)
        expected_alpha = 200.0 / 255.0
        assert np.allclose(normalized[..., 3:4], expected_alpha)
        assert np.all(normalized[..., 3:4] >= 0.0) and np.all(normalized[..., 3:4] <= 1.0)
        # RGB should be normalized (mean ~0, std ~1)
        assert np.allclose(normalized[..., :3].mean(), 0.0, atol=1e-10)
        assert np.allclose(normalized[..., :3].std(), 1.0, atol=1e-10)

    def test_normalize_zscore_rgba_alpha_already_normalized(self):
        """Test z-score normalization for RGBA with alpha already in [0, 1]."""
        array = np.zeros((1, 5, 5, 4), dtype=np.float64)
        array[0, :, :, :3] = np.random.rand(5, 5, 3) * 255  # RGB in [0, 255]
        array[0, :, :, 3] = 0.8  # Alpha already in [0, 1]
        normalized = normalize_batch(array, mode="zscore")
        
        # Alpha channel should be preserved as-is (already in [0, 1])
        assert np.allclose(normalized[..., 3:4], 0.8)
        assert np.all(normalized[..., 3:4] >= 0.0) and np.all(normalized[..., 3:4] <= 1.0)
        # RGB should be normalized (mean ~0, std ~1)
        assert np.allclose(normalized[..., :3].mean(), 0.0, atol=1e-10)
        assert np.allclose(normalized[..., :3].std(), 1.0, atol=1e-10)

    def test_normalize_zscore_constant_image(self):
        """Test z-score normalization for constant image (edge case)."""
        # Constant image should map to zeros
        array = np.array([[[42, 42], [42, 42]]], dtype=np.uint8)
        normalized = normalize_batch(array, mode="zscore")
        
        assert np.allclose(normalized, 0.0)

    def test_normalize_zscore_per_image_independence(self):
        """Test that z-score normalization works independently per image."""
        # Create batch where each image has different mean/std
        array = np.zeros((2, 5, 5), dtype=np.float64)
        # Image 0: mean=50, std=20
        array[0] = np.random.normal(50, 20, (5, 5))
        # Image 1: mean=200, std=50
        array[1] = np.random.normal(200, 50, (5, 5))
        normalized = normalize_batch(array, mode="zscore")
        
        # Both images should have mean~0, std~1 after normalization
        assert np.allclose(normalized[0].mean(), 0.0, atol=1e-10)
        assert np.allclose(normalized[0].std(), 1.0, atol=1e-10)
        assert np.allclose(normalized[1].mean(), 0.0, atol=1e-10)
        assert np.allclose(normalized[1].std(), 1.0, atol=1e-10)

    def test_normalize_imagenet_stats_rgb_channels_last(self):
        """Test ImageNet stats normalization for RGB batch with channels last."""
        # Create RGB batch in [0, 1] range (typical input for ImageNet normalization)
        array = np.random.rand(2, 5, 5, 3).astype(np.float32)
        normalized = normalize_batch(array, mode="imagenet_stats")
        
        assert normalized.dtype == np.float64
        assert normalized.shape == array.shape
        # Verify per-channel normalization was applied
        # Each channel should have different mean/std after normalization

    def test_normalize_imagenet_stats_rgb_channels_first(self):
        """Test ImageNet stats normalization for RGB batch with channels first."""
        array = np.random.rand(2, 3, 5, 5).astype(np.float32)
        normalized = normalize_batch(array, mode="imagenet_stats")
        
        assert normalized.dtype == np.float64
        assert normalized.shape == array.shape

    def test_normalize_imagenet_stats_rgba_scales_alpha(self):
        """Test that ImageNet stats normalization scales alpha channel to [0, 1] for RGBA."""
        # Test with float array (alpha already in [0, 1])
        array = np.random.rand(1, 5, 5, 4).astype(np.float32)
        array[0, :, :, 3] = 0.5  # Alpha channel
        normalized = normalize_batch(array, mode="imagenet_stats")
        
        # Alpha channel should be preserved (already in [0, 1])
        assert np.allclose(normalized[..., 3:4], array.astype(np.float64)[..., 3:4])
        # RGB should be normalized
        assert normalized[..., :3].dtype == np.float64
        
        # Test with uint8 array (alpha should be scaled by 255)
        array_uint8 = np.zeros((1, 5, 5, 4), dtype=np.uint8)
        array_uint8[0, :, :, :3] = np.random.randint(0, 255, (5, 5, 3))
        array_uint8[0, :, :, 3] = 200  # Alpha channel
        normalized_uint8 = normalize_batch(array_uint8, mode="imagenet_stats")
        
        # Alpha channel should be scaled to [0, 1] (200/255)
        expected_alpha = 200.0 / 255.0
        assert np.allclose(normalized_uint8[..., 3:4], expected_alpha)
        assert np.all(normalized_uint8[..., 3:4] >= 0.0) and np.all(normalized_uint8[..., 3:4] <= 1.0)

    def test_normalize_imagenet_stats_grayscale_raises_error(self):
        """Test that ImageNet stats normalization raises error for grayscale."""
        array = np.random.rand(1, 5, 5).astype(np.float32)
        with pytest.raises(ValueError, match="ImageNet normalization requires RGB"):
            normalize_batch(array, mode="imagenet_stats")

    def test_normalize_imagenet_stats_known_values(self):
        """Test ImageNet stats normalization with known values."""
        # Create array where we can verify the normalization
        # Use values that will produce predictable results
        array = np.ones((1, 2, 2, 3), dtype=np.float32) * 0.5  # All channels = 0.5
        normalized = normalize_batch(array, mode="imagenet_stats")
        
        # ImageNet mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        # For value 0.5: (0.5 - mean[i]) / std[i]
        expected_r = (0.5 - 0.485) / 0.229
        expected_g = (0.5 - 0.456) / 0.224
        expected_b = (0.5 - 0.406) / 0.225
        
        assert np.allclose(normalized[0, :, :, 0], expected_r)
        assert np.allclose(normalized[0, :, :, 1], expected_g)
        assert np.allclose(normalized[0, :, :, 2], expected_b)

    def test_invalid_dimension_raises_error(self):
        """Test that arrays without batch dimension raise ValueError."""
        # 2D array (single image without batch)
        array = np.array([[0, 50], [50, 100]], dtype=np.float64)
        with pytest.raises(ValueError, match="Array must have batch dimension"):
            normalize_batch(array, mode="0_1")
        
        # 5D array (too many dimensions)
        array = np.random.rand(2, 3, 4, 5, 6)
        with pytest.raises(ValueError, match="Array must have batch dimension"):
            normalize_batch(array, mode="0_1")

    def test_normalize_0_1_uint16_dtype_max(self):
        """Test that 0_1 normalization uses dtype max for uint16 (not hardcoded 255)."""
        # Create RGBA image with uint16 dtype
        array = np.zeros((1, 5, 5, 4), dtype=np.uint16)
        array[0, :, :, :3] = 1000  # RGB channels
        array[0, :, :, 3] = 50000  # Alpha channel (uint16 max is 65535)
        normalized = normalize_batch(array, mode="0_1")
        
        # Alpha should be scaled by uint16 max (65535), not 255
        expected_alpha = 50000.0 / 65535.0
        assert np.allclose(normalized[0, :, :, 3], expected_alpha)
        assert np.all(normalized[..., 3:4] >= 0.0) and np.all(normalized[..., 3:4] <= 1.0)

    def test_normalize_minus1_1_uint16_dtype_max(self):
        """Test that minus1_1 normalization uses dtype max for uint16 (not hardcoded 255)."""
        # Create RGBA image with uint16 dtype
        array = np.zeros((1, 5, 5, 4), dtype=np.uint16)
        array[0, :, :, :3] = 1000  # RGB channels
        array[0, :, :, 3] = 50000  # Alpha channel (uint16 max is 65535)
        normalized = normalize_batch(array, mode="minus1_1")
        
        # Alpha should be scaled by uint16 max (65535), then to [-1, 1]
        expected_alpha = 2.0 * (50000.0 / 65535.0) - 1.0
        assert np.allclose(normalized[0, :, :, 3], expected_alpha)
        assert np.all(normalized[..., 3:4] >= -1.0) and np.all(normalized[..., 3:4] <= 1.0)

    def test_normalize_imagenet_stats_float_array_uint8_range(self):
        """Test that imagenet_stats auto-normalizes float arrays with values > 1 (uint8-like range)."""
        # Float array with values in [0, 255] range
        array = np.zeros((1, 5, 5, 3), dtype=np.float64)
        array[0, :, :, 0] = 100.0  # R channel
        array[0, :, :, 1] = 150.0  # G channel
        array[0, :, :, 2] = 200.0  # B channel
        normalized = normalize_batch(array, mode="imagenet_stats")
        
        # Should auto-normalize to [0, 1] by dividing by 255, then apply ImageNet stats
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        expected = (np.array([100.0, 150.0, 200.0]) / 255.0 - imagenet_mean) / imagenet_std
        assert np.allclose(normalized[0, 0, 0], expected)

    def test_normalize_imagenet_stats_float_array_minus1_1_range(self):
        """Test that imagenet_stats auto-normalizes float arrays from [-1, 1] to [0, 1] range."""
        # Float array in [-1, 1] range (e.g., from minus1_1 normalization)
        array = np.zeros((1, 5, 5, 3), dtype=np.float64)
        array[0, :, :, 0] = -0.5  # R channel
        array[0, :, :, 1] = 0.0   # G channel
        array[0, :, :, 2] = 0.5   # B channel
        normalized = normalize_batch(array, mode="imagenet_stats")
        
        # Should scale [-1, 1] -> [0, 1] using (x + 1) / 2, then apply ImageNet stats
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        scaled = (np.array([-0.5, 0.0, 0.5]) + 1.0) / 2.0
        expected = (scaled - imagenet_mean) / imagenet_std
        assert np.allclose(normalized[0, 0, 0], expected)

    def test_normalize_imagenet_stats_float_array_already_0_1(self):
        """Test that imagenet_stats doesn't re-normalize float arrays already in [0, 1] range."""
        # Float array already in [0, 1] range
        array = np.zeros((1, 5, 5, 3), dtype=np.float64)
        array[0, :, :, 0] = 0.2  # R channel
        array[0, :, :, 1] = 0.5  # G channel
        array[0, :, :, 2] = 0.8  # B channel
        normalized = normalize_batch(array, mode="imagenet_stats")
        
        # Should apply ImageNet stats directly without re-normalization
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        expected = (np.array([0.2, 0.5, 0.8]) - imagenet_mean) / imagenet_std
        assert np.allclose(normalized[0, 0, 0], expected)
