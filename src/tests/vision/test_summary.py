import pytest
import numpy as np

from fenn.vision.summary import image_summary


class TestImageSummary:
    """Test suite for image_summary function."""

    def test_3d_grayscale_batch(self):
        """Test summary for 3D grayscale batch (N, H, W)."""
        array = np.random.randint(0, 255, (10, 224, 224), dtype=np.uint8)
        result = image_summary(array)
        
        assert result["is_grayscale"] is True
        assert result["channel_location"] is None
        assert result["batch_size"] == 10
        assert result["shape_info"]["height"] == 224
        assert result["shape_info"]["width"] == 224
        assert result["shape_info"]["channels"] == 1
        assert result["dtype"]["name"] == "uint8"
        assert "min" in result["value_range"]
        assert "max" in result["value_range"]
        assert len(result["channel_stats"]["mean"]) == 1
        assert len(result["channel_stats"]["std"]) == 1
        assert result["data_quality"]["has_nan"] is False
        assert result["data_quality"]["has_inf"] is False

    def test_4d_channels_last_color(self):
        """Test summary for 4D color batch with channels last (N, H, W, C)."""
        array = np.random.randint(0, 255, (32, 224, 224, 3), dtype=np.uint8)
        result = image_summary(array)
        
        assert result["is_grayscale"] is False
        assert result["channel_location"] == "last"
        assert result["batch_size"] == 32
        assert result["shape_info"]["height"] == 224
        assert result["shape_info"]["width"] == 224
        assert result["shape_info"]["channels"] == 3
        assert len(result["channel_stats"]["mean"]) == 3
        assert len(result["channel_stats"]["std"]) == 3

    def test_4d_channels_first_color(self):
        """Test summary for 4D color batch with channels first (N, C, H, W)."""
        array = np.random.rand(16, 3, 128, 128).astype(np.float32)
        result = image_summary(array)
        
        assert result["is_grayscale"] is False
        assert result["channel_location"] == "first"
        assert result["batch_size"] == 16
        assert result["shape_info"]["channels"] == 3
        assert result["shape_info"]["height"] == 128
        assert result["shape_info"]["width"] == 128
        assert result["dtype"]["name"] == "float32"
        assert len(result["channel_stats"]["mean"]) == 3

    def test_4d_grayscale_with_channel_dimension(self):
        """Test summary for 4D grayscale batch (N, H, W, 1)."""
        array = np.random.randint(0, 255, (5, 64, 64, 1), dtype=np.uint8)
        result = image_summary(array)
        
        assert result["is_grayscale"] is True
        assert result["channel_location"] == "last"
        assert result["shape_info"]["channels"] == 1
        assert len(result["channel_stats"]["mean"]) == 1

    def test_with_nan_values(self):
        """Test summary handles NaN values correctly."""
        array = np.random.rand(8, 224, 224, 3).astype(np.float32)
        array[0, 0, 0, 0] = np.nan  # Introduce NaN
        result = image_summary(array)
        
        assert result["data_quality"]["has_nan"] is True
        assert result["data_quality"]["nan_count"] > 0
        # Should still compute stats using nan-aware functions
        assert len(result["channel_stats"]["mean"]) == 3
        assert all(not np.isnan(m) for m in result["channel_stats"]["mean"])

    def test_computed_statistics_accuracy(self):
        """Test that min, max, mean, and std values are computed correctly."""
        # Create array with known values for channels last format (N, H, W, C)
        array = np.zeros((2, 10, 10, 3), dtype=np.float32)
        # Image 1: set known values in first 2x2 region
        array[0, 0, 0, :] = [10, 20, 30]
        array[0, 0, 1, :] = [40, 50, 60]
        array[0, 1, 0, :] = [70, 80, 90]
        array[0, 1, 1, :] = [100, 110, 120]
        # Image 2: set known values in first 2x2 region
        array[1, 0, 0, :] = [5, 15, 25]
        array[1, 0, 1, :] = [35, 45, 55]
        array[1, 1, 0, :] = [65, 75, 85]
        array[1, 1, 1, :] = [95, 105, 115]
        
        result = image_summary(array)
        
        # Verify value range
        assert result["value_range"]["min"] == 0.0
        assert result["value_range"]["max"] == 120.0
        
        channel_0_values = array[:, :, :, 0].flatten()  # All values in channel 0
        channel_1_values = array[:, :, :, 1].flatten()  # All values in channel 1
        channel_2_values = array[:, :, :, 2].flatten()  # All values in channel 2

        expected_means = [
            float(np.nanmean(channel_0_values)),
            float(np.nanmean(channel_1_values)),
            float(np.nanmean(channel_2_values))
        ]
        np.testing.assert_allclose(result["channel_stats"]["mean"], expected_means, rtol=1e-5)
        
        expected_stds = [
            float(np.nanstd(channel_0_values)),
            float(np.nanstd(channel_1_values)),
            float(np.nanstd(channel_2_values))
        ]
        np.testing.assert_allclose(result["channel_stats"]["std"], expected_stds, rtol=1e-5)

