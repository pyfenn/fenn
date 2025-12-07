import numpy as np
from typing import TypedDict, Literal, Any, Dict

from fenn.vision.vision_utils import detect_format


def _extract_shape_info(
    shape: tuple, 
    channel_location: Literal["first", "last"] | None
) -> Dict[str, Any]:
    """
    Extract shape information (H, W, C) from array shape based on channel location.
    
    Args:
        shape: Array shape tuple
        channel_location: Channel position ("first", "last", or None)
    
    Returns:
        Dictionary with height, width, channels, and full_shape
    """
    if channel_location == "last":
        # (N, H, W, C) - channels last (color or grayscale with C=1)
        height = shape[1]
        width = shape[2]
        channels = shape[3]
    elif channel_location == "first":
        # (N, C, H, W) - channels first (color or grayscale with C=1)
        channels = shape[1]
        height = shape[2]
        width = shape[3]
    else:
        # (N, H, W) - grayscale without channel dimension
        height = shape[1]
        width = shape[2]
        channels = 1
    
    return {
        "height": height,
        "width": width,
        "channels": channels,
        "full_shape": tuple(shape)
    }


class ImageSummary(TypedDict):
    """Summary information for an image batch."""
    is_grayscale: bool
    channel_location: Literal["first", "last"] | None
    batch_size: int
    shape_info: Dict[str, Any]  # Contains: height, width, channels, full_shape
    dtype: Dict[str, Any]  # Contains: name, kind, itemsize
    value_range: Dict[str, Any]  # TODO: Define structure for value_range
    channel_stats: Dict[str, Any]  # TODO: Define structure for channel_stats
    data_quality: Dict[str, Any]  # TODO: Define structure for data_quality


def image_summary(array: np.ndarray) -> ImageSummary:
    """
    Generate a comprehensive summary of an image batch or dataset.
    
    Provides a one-shot overview combining shape distribution (H, W, C), dtype,
    value range, per-channel mean/std, and simple NaN/inf checks. Useful for
    early-stage exploratory data analysis and sanity-checking image datasets.

    Args:
        array: Image array in batch format. The first dimension (N) must represent
            the batch size. Supported formats:
            - (N, H, W, C) - batch of images with channels last
            - (N, C, H, W) - batch of images with channels first
            - (N, H, W) - batch of grayscale images
            
            Note: For single images, wrap with array[np.newaxis, ...] to add batch dimension.
            All images in the batch must have consistent dimensions (H, W, C). For variable-sized
            images, use resize_batch() first to normalize them to a uniform size.
    
    Returns:
        Dictionary containing:
            - shape_info: Shape information (H, W, C) for the batch.
            - dtype: Data type of the array
            - value_range: Min and max values in the array
            - channel_stats: Per-channel mean and standard deviation
            - data_quality: Checks for NaN and Inf values
            - is_grayscale: bool - True if grayscale images
            - channel_location: str | None - "first", "last", or None for grayscale
            - batch_size: Number of images in the batch (N)
    
    Raises:
        ValueError: If array is not a valid image array format
        TypeError: If array is not a numpy array
    
    Example:
        >>> import numpy as np
        >>> from fenn.vision import image_summary
        >>> # Batch of images
        >>> images = np.random.randint(0, 255, (32, 224, 224, 3), dtype=np.uint8)
        >>> summary = image_summary(images)
        >>> print(summary['batch_size'])  # 32
        >>> # Single image (wrap with batch dimension)
        >>> single = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        >>> summary = image_summary(single[np.newaxis, ...])
        >>> print(summary['batch_size'])  # 1
    """
    # Validate and detect format
    format_info = detect_format(array)
    is_grayscale = format_info["is_grayscale"]
    channel_location = format_info["channel_location"]
    batch_size = array.shape[0]
    
    # Extract shape information (H, W, C)
    shape_info = _extract_shape_info(array.shape, channel_location)
    
    # Extract dtype information
    dtype_info = {
        "name": str(array.dtype),
        "kind": array.dtype.kind,  # 'i' for int, 'f' for float, 'u' for unsigned int, etc.
        "itemsize": array.dtype.itemsize  # Size in bytes
    }
    
    # Check for NaN and Inf values
    has_nan = bool(np.isnan(array).any())
    has_inf = bool(np.isinf(array).any())
    nan_count = int(np.isnan(array).sum()) if has_nan else 0
    inf_count = int(np.isinf(array).sum()) if has_inf else 0
    
    data_quality = {
        "has_nan": has_nan,
        "has_inf": has_inf,
        "nan_count": nan_count,
        "inf_count": inf_count
    }
    
    # Calculate value range (min, max) across entire array
    value_range = {
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array))
    }
    
    # Calculate per-channel mean and std
    if channel_location == "last":
        # (N, H, W, C) - compute mean/std along batch, height, width axes
        channel_means = np.nanmean(array, axis=(0, 1, 2)).tolist()
        channel_stds = np.nanstd(array, axis=(0, 1, 2)).tolist()
    elif channel_location == "first":
        # (N, C, H, W) - compute mean/std along batch, height, width axes
        channel_means = np.nanmean(array, axis=(0, 2, 3)).tolist()
        channel_stds = np.nanstd(array, axis=(0, 2, 3)).tolist()
    else:
        # (N, H, W) - grayscale without channel dimension, single value
        channel_means = [float(np.nanmean(array))]
        channel_stds = [float(np.nanstd(array))]
    
    channel_stats = {
        "mean": channel_means,
        "std": channel_stds
    }
    
    return {
        "is_grayscale": is_grayscale,
        "channel_location": channel_location,
        "batch_size": batch_size,
        "shape_info": shape_info,
        "dtype": dtype_info,
        "value_range": value_range,
        "channel_stats": channel_stats,
        "data_quality": data_quality
    }
