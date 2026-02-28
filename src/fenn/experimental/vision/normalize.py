import numpy as np
from typing import Literal

from .vision_utils import detect_format


def normalize_batch(
    array: np.ndarray,
    mode: Literal["0_1", "minus1_1", "imagenet_stats", "zscore"] = "0_1",
) -> np.ndarray:
    """
    Normalize a batch of images using the specified normalization mode.
    
    Args:
        array: Input image batch array. Must have batch dimension as first dimension:
            - (N, H, W) - batch of grayscale images
            - (N, H, W, C) - batch with channels last
            - (N, C, H, W) - batch with channels first
        mode: Normalization mode to apply:
            - "0_1": Scale values to [0, 1] range
            - "minus1_1": Scale values to [-1, 1] range
            - "imagenet_stats": Normalize using ImageNet statistics (mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]) for RGB channels
            - "zscore": Standardize using z-score normalization (mean=0, std=1)
    
    Returns:
        Normalized array with same shape as input, float64 dtype.
    
    Raises:
        TypeError: If array is not a numpy array
        ValueError: If array doesn't have batch dimension or has unsupported format
        ValueError: If mode is not one of the supported modes
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(array)}")

    if array.ndim < 3 or array.ndim > 4:
        raise ValueError(
            f"Array must have batch dimension. Expected 3D greyscale (N, H, W) or 4D color "
            f"(N, H, W, C) / (N, C, H, W), got {array.ndim}D array with shape {array.shape}. "
            f"For single images, wrap with array[np.newaxis, ...] to add batch dimension."
        )
    
    if mode == "0_1":
        return _normalize_0_1(array)
    elif mode == "minus1_1":
        return _normalize_minus1_1(array)
    elif mode == "imagenet_stats":
        return _normalize_imagenet_stats(array)
    elif mode == "zscore":
        return _normalize_zscore(array)
    else:
        raise ValueError(
            f"Unsupported normalization mode: {mode}. "
            f"Expected one of: '0_1', 'minus1_1', 'imagenet_stats', 'zscore'"
        )


def _normalize_0_1(array: np.ndarray) -> np.ndarray:
    """
    Normalize array to [0, 1] range.
    
    Normalizes each image in the batch independently by scaling values so that
    the minimum becomes 0 and the maximum becomes 1. Formula: (x - min) / (max - min).
    
    If an image has constant values (min == max), it will be set to 0.5.
    For RGBA images, alpha channel is scaled by dtype max (not min/max normalized).
    
    Args:
        array: Input array with batch dimension
    
    Returns:
        Normalized array in [0, 1] range with float64 dtype
    """
    # Convert to float64 to ensure we can represent values in [0, 1]
    array_float = array.astype(np.float64)

    format_info = detect_format(array)
    channel_location = format_info["channel_location"]
    
    # Check if this is RGBA (4 channels)
    is_rgba = False
    if channel_location == "last" and array.shape[3] == 4:
        is_rgba = True
    elif channel_location == "first" and array.shape[1] == 4:
        is_rgba = True
    
    if is_rgba:
        # For RGBA: normalize RGB using min/max, scale alpha by dtype max
        if channel_location == "last":
            rgb_channels = array_float[..., :3]
            alpha_channel = array_float[..., 3:4]
            
            # Normalize RGB using min/max
            min_vals = np.min(rgb_channels, axis=(1, 2), keepdims=True)
            max_vals = np.max(rgb_channels, axis=(1, 2), keepdims=True)
            range_vals = max_vals - min_vals
            normalized_rgb = np.where(
                range_vals > 0,
                (rgb_channels - min_vals) / range_vals,
                0.5
            )
            
            # Scale alpha by dtype max
            if np.issubdtype(array.dtype, np.integer):
                dtype_max = np.iinfo(array.dtype).max
                normalized_alpha = alpha_channel / dtype_max
            else:
                # Float type: if > 1, assume [0, 255] range, else already normalized
                if alpha_channel.max() > 1.0:
                    normalized_alpha = alpha_channel / 255.0
                else:
                    normalized_alpha = alpha_channel
            
            return np.concatenate([normalized_rgb, normalized_alpha], axis=-1)
        else:  # channels first
            rgb_channels = array_float[:, :3, ...]
            alpha_channel = array_float[:, 3:4, ...]
            
            # Normalize RGB using min/max
            min_vals = np.min(rgb_channels, axis=(2, 3), keepdims=True)
            max_vals = np.max(rgb_channels, axis=(2, 3), keepdims=True)
            range_vals = max_vals - min_vals
            normalized_rgb = np.where(
                range_vals > 0,
                (rgb_channels - min_vals) / range_vals,
                0.5
            )
            
            # Scale alpha by dtype max
            if np.issubdtype(array.dtype, np.integer):
                dtype_max = np.iinfo(array.dtype).max
                normalized_alpha = alpha_channel / dtype_max
            else:
                # Float type: if > 1, assume [0, 255] range, else already normalized
                if alpha_channel.max() > 1.0:
                    normalized_alpha = alpha_channel / 255.0
                else:
                    normalized_alpha = alpha_channel
            
            return np.concatenate([normalized_rgb, normalized_alpha], axis=1)
    
    # Normalize all channels using the same strategy (non-RGBA case)
    if channel_location == "first":
        min_vals = np.min(array_float, axis=(2, 3), keepdims=True)
        max_vals = np.max(array_float, axis=(2, 3), keepdims=True)
    else:  # channel_location is None or "last"
        min_vals = np.min(array_float, axis=(1, 2), keepdims=True)
        max_vals = np.max(array_float, axis=(1, 2), keepdims=True)
    
    range_vals = max_vals - min_vals
    return np.where(
        range_vals > 0,
        (array_float - min_vals) / range_vals,
        0.5
    )


def _normalize_minus1_1(array: np.ndarray) -> np.ndarray:
    """
    Normalize array to [-1, 1] range.
    
    Normalizes each image in the batch independently by scaling values so that
    the minimum becomes -1 and the maximum becomes 1. Formula: 2 * (x - min) / (max - min) - 1.
    
    If an image has constant values (min == max), it will be set to 0.0.
    For RGBA images, alpha channel is scaled by dtype max, then mapped to [-1, 1].
    
    Args:
        array: Input array with batch dimension
    
    Returns:
        Normalized array in [-1, 1] range with float64 dtype
    """
    # Convert to float64 to ensure we can represent values in [-1, 1]
    array_float = array.astype(np.float64)

    format_info = detect_format(array)
    channel_location = format_info["channel_location"]
    
    # Check if this is RGBA (4 channels)
    is_rgba = False
    if channel_location == "last" and array.shape[3] == 4:
        is_rgba = True
    elif channel_location == "first" and array.shape[1] == 4:
        is_rgba = True
    
    if is_rgba:
        # For RGBA: normalize RGB using min/max, scale alpha by dtype max then to [-1, 1]
        if channel_location == "last":
            rgb_channels = array_float[..., :3]
            alpha_channel = array_float[..., 3:4]
            
            # Normalize RGB using min/max
            min_vals = np.min(rgb_channels, axis=(1, 2), keepdims=True)
            max_vals = np.max(rgb_channels, axis=(1, 2), keepdims=True)
            range_vals = max_vals - min_vals
            normalized_rgb = np.where(
                range_vals > 0,
                2.0 * (rgb_channels - min_vals) / range_vals - 1.0,
                0.0
            )
            
            # Scale alpha by dtype max, then to [-1, 1]
            if np.issubdtype(array.dtype, np.integer):
                dtype_max = np.iinfo(array.dtype).max
                normalized_alpha = 2.0 * (alpha_channel / dtype_max) - 1.0
            else:
                # Float type: if > 1, assume [0, 255] range, else already in [0, 1]
                if alpha_channel.max() > 1.0:
                    normalized_alpha = 2.0 * (alpha_channel / 255.0) - 1.0
                else:
                    # Already in [0, 1], map to [-1, 1]
                    normalized_alpha = 2.0 * alpha_channel - 1.0
            
            return np.concatenate([normalized_rgb, normalized_alpha], axis=-1)
        else:  # channels first
            rgb_channels = array_float[:, :3, ...]
            alpha_channel = array_float[:, 3:4, ...]
            
            # Normalize RGB using min/max
            min_vals = np.min(rgb_channels, axis=(2, 3), keepdims=True)
            max_vals = np.max(rgb_channels, axis=(2, 3), keepdims=True)
            range_vals = max_vals - min_vals
            normalized_rgb = np.where(
                range_vals > 0,
                2.0 * (rgb_channels - min_vals) / range_vals - 1.0,
                0.0
            )
            
            # Scale alpha by dtype max, then to [-1, 1]
            if np.issubdtype(array.dtype, np.integer):
                dtype_max = np.iinfo(array.dtype).max
                normalized_alpha = 2.0 * (alpha_channel / dtype_max) - 1.0
            else:
                # Float type: if > 1, assume [0, 255] range, else already in [0, 1]
                if alpha_channel.max() > 1.0:
                    normalized_alpha = 2.0 * (alpha_channel / 255.0) - 1.0
                else:
                    # Already in [0, 1], map to [-1, 1]
                    normalized_alpha = 2.0 * alpha_channel - 1.0
            
            return np.concatenate([normalized_rgb, normalized_alpha], axis=1)
    
    # Normalize all channels using the same strategy (non-RGBA case)
    if channel_location == "first":
        min_vals = np.min(array_float, axis=(2, 3), keepdims=True)
        max_vals = np.max(array_float, axis=(2, 3), keepdims=True)
    else:  # channel_location is None or "last"
        min_vals = np.min(array_float, axis=(1, 2), keepdims=True)
        max_vals = np.max(array_float, axis=(1, 2), keepdims=True)
    
    range_vals = max_vals - min_vals
    return np.where(
        range_vals > 0,
        2.0 * (array_float - min_vals) / range_vals - 1.0,
        0.0
    )


def _normalize_imagenet_stats(array: np.ndarray) -> np.ndarray:
    """
    Normalize array using ImageNet statistics.
    
    Uses mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] for RGB channels (standard imagenet values).
    For RGBA images, alpha channel is scaled to [0, 1] by dtype max (not normalized with ImageNet stats).
    Assumes RGB format with channels last (N, H, W, 3) or channels first (N, 3, H, W).
    Formula: (x - mean) / std, applied per channel.
    
    Automatically normalizes input to [0, 1] range first if values are > 1 (e.g., uint8 [0, 255]),
    then applies ImageNet statistics. This makes it consistent with other normalization modes.
    
    Args:
        array: Input array with batch dimension
    
    Returns:
        Normalized array using ImageNet statistics, float64 dtype
    """
    # Convert to float64 for precision
    array_float = array.astype(np.float64)

    format_info = detect_format(array)
    channel_location = format_info["channel_location"]
    
    # ImageNet statistics (per RGB channel)
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float64)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float64)
    
    # Check if this is RGBA (4 channels)
    is_rgba = False
    if channel_location == "last" and array.shape[3] == 4:
        is_rgba = True
    elif channel_location == "first" and array.shape[1] == 4:
        is_rgba = True
    
    # For RGBA, extract alpha before auto-normalization to handle separately
    if is_rgba:
        if channel_location == "last":
            rgb_channels_float = array_float[..., :3]
            alpha_channel_original = array_float[..., 3:4].copy()
        else:  # channels first
            rgb_channels_float = array_float[:, :3, ...]
            alpha_channel_original = array_float[:, 3:4, ...].copy()
        
        # Auto-normalize only RGB channels to [0, 1] if needed
        rgb_max = rgb_channels_float.max()
        rgb_min = rgb_channels_float.min()
        
        if rgb_min < 0.0:
            if rgb_max <= 1.0:
                rgb_channels_float = (rgb_channels_float + 1.0) / 2.0
            else:
                rgb_channels_float = rgb_channels_float / rgb_max
        elif rgb_max > 1.0:
            if np.issubdtype(array.dtype, np.integer):
                dtype_max = np.iinfo(array.dtype).max
                rgb_channels_float = rgb_channels_float / dtype_max
            else:
                rgb_channels_float = rgb_channels_float / 255.0
    else:
        # Auto-normalize entire array to [0, 1] if values are outside expected [0, 1] range
        array_max = array_float.max()
        array_min = array_float.min()
        
        if array_min < 0.0:
            # Values are < 0, likely in [-1, 1] range
            if array_max <= 1.0:
                # Scale from [-1, 1] to [0, 1]: (x + 1) / 2
                array_float = (array_float + 1.0) / 2.0
            else:
                # If max > 1 and min < 0, weird range - just divide by max
                array_float = array_float / array_max
        elif array_max > 1.0:
            # Values are > 1, need to normalize
            if np.issubdtype(array.dtype, np.integer):
                # Integer types: divide by dtype max (e.g., uint8 -> divide by 255)
                dtype_max = np.iinfo(array.dtype).max
                array_float = array_float / dtype_max
            else:
                # Float types: assume [0, 255] range and divide by 255
                array_float = array_float / 255.0
    
    if is_rgba:
        # For RGBA: apply ImageNet stats to RGB, scale alpha to [0, 1] by dtype max
        # Apply ImageNet stats to RGB
        if channel_location == "last":
            mean_broadcast = imagenet_mean.reshape(1, 1, 1, 3)
            std_broadcast = imagenet_std.reshape(1, 1, 1, 3)
            normalized_rgb = (rgb_channels_float - mean_broadcast) / std_broadcast
            
            # Scale alpha to [0, 1] by dtype max
            if np.issubdtype(array.dtype, np.integer):
                dtype_max = np.iinfo(array.dtype).max
                normalized_alpha = alpha_channel_original / dtype_max
            else:
                # Float type: if > 1, assume [0, 255] range, else already in [0, 1]
                if alpha_channel_original.max() > 1.0:
                    normalized_alpha = alpha_channel_original / 255.0
                else:
                    # Already in [0, 1], preserve as-is
                    normalized_alpha = alpha_channel_original
            
            return np.concatenate([normalized_rgb, normalized_alpha], axis=-1)
        else:  # channels first
            mean_broadcast = imagenet_mean.reshape(1, 3, 1, 1)
            std_broadcast = imagenet_std.reshape(1, 3, 1, 1)
            normalized_rgb = (rgb_channels_float - mean_broadcast) / std_broadcast
            
            # Scale alpha to [0, 1] by dtype max
            if np.issubdtype(array.dtype, np.integer):
                dtype_max = np.iinfo(array.dtype).max
                normalized_alpha = alpha_channel_original / dtype_max
            else:
                # Float type: if > 1, assume [0, 255] range, else already in [0, 1]
                if alpha_channel_original.max() > 1.0:
                    normalized_alpha = alpha_channel_original / 255.0
                else:
                    # Already in [0, 1], preserve as-is
                    normalized_alpha = alpha_channel_original
            
            return np.concatenate([normalized_rgb, normalized_alpha], axis=1)
    
    # For RGB: apply ImageNet stats
    if channel_location == "last":
        # (N, H, W, 3) - broadcast mean/std to (1, 1, 1, 3)
        mean_broadcast = imagenet_mean.reshape(1, 1, 1, 3)
        std_broadcast = imagenet_std.reshape(1, 1, 1, 3)
        return (array_float - mean_broadcast) / std_broadcast
    elif channel_location == "first":
        # (N, 3, H, W) - broadcast mean/std to (1, 3, 1, 1)
        mean_broadcast = imagenet_mean.reshape(1, 3, 1, 1)
        std_broadcast = imagenet_std.reshape(1, 3, 1, 1)
        return (array_float - mean_broadcast) / std_broadcast
    else:
        # Grayscale (N, H, W) - ImageNet stats don't apply, raise error
        raise ValueError(
            "ImageNet normalization requires RGB images. "
            f"Got grayscale array with shape {array.shape}. "
            "Use ensure_color_mode() to convert to RGB first."
        )


def _normalize_zscore(array: np.ndarray) -> np.ndarray:
    """
    Normalize array using z-score normalization (mean=0, std=1).
    
    Normalizes each image in the batch independently by standardizing values so that
    the mean becomes 0 and the standard deviation becomes 1. Formula: (x - mean) / std.
    
    If an image has constant values (std == 0), it will be set to zeros.
    For RGBA images, alpha channel is scaled to [0, 1] by dtype max (not z-score normalized).
    
    Args:
        array: Input array with batch dimension
    
    Returns:
        Standardized array with mean=0 and std=1, float64 dtype
    """
    # Convert to float64 for precision in mean/std calculations
    array_float = array.astype(np.float64)

    format_info = detect_format(array)
    channel_location = format_info["channel_location"]
    
    # Check if this is RGBA (4 channels)
    is_rgba = False
    if channel_location == "last" and array.shape[3] == 4:
        is_rgba = True
    elif channel_location == "first" and array.shape[1] == 4:
        is_rgba = True
    
    if is_rgba:
        # For RGBA: z-score normalize RGB, scale alpha to [0, 1] by dtype max
        if channel_location == "last":
            rgb_channels = array_float[..., :3]
            alpha_channel = array_float[..., 3:4]
            
            # Z-score normalize RGB
            mean_vals = np.mean(rgb_channels, axis=(1, 2), keepdims=True)
            std_vals = np.std(rgb_channels, axis=(1, 2), keepdims=True)
            normalized_rgb = np.where(
                std_vals > 0,
                (rgb_channels - mean_vals) / std_vals,
                0.0
            )
            
            # Scale alpha to [0, 1] by dtype max (if integer) or preserve if already normalized
            if np.issubdtype(array.dtype, np.integer):
                dtype_max = np.iinfo(array.dtype).max
                normalized_alpha = alpha_channel / dtype_max
            else:
                # Float type: if > 1, assume [0, 255] range, else already in [0, 1]
                if alpha_channel.max() > 1.0:
                    normalized_alpha = alpha_channel / 255.0
                else:
                    # Already in [0, 1], preserve as-is
                    normalized_alpha = alpha_channel
            
            return np.concatenate([normalized_rgb, normalized_alpha], axis=-1)
        else:  # channels first
            rgb_channels = array_float[:, :3, ...]
            alpha_channel = array_float[:, 3:4, ...]
            
            # Z-score normalize RGB
            mean_vals = np.mean(rgb_channels, axis=(2, 3), keepdims=True)
            std_vals = np.std(rgb_channels, axis=(2, 3), keepdims=True)
            normalized_rgb = np.where(
                std_vals > 0,
                (rgb_channels - mean_vals) / std_vals,
                0.0
            )
            
            # Scale alpha to [0, 1] by dtype max (if integer) or preserve if already normalized
            if np.issubdtype(array.dtype, np.integer):
                dtype_max = np.iinfo(array.dtype).max
                normalized_alpha = alpha_channel / dtype_max
            else:
                # Float type: if > 1, assume [0, 255] range, else already in [0, 1]
                if alpha_channel.max() > 1.0:
                    normalized_alpha = alpha_channel / 255.0
                else:
                    # Already in [0, 1], preserve as-is
                    normalized_alpha = alpha_channel
            
            return np.concatenate([normalized_rgb, normalized_alpha], axis=1)
    
    # Normalize all channels using the same strategy (non-RGBA case)
    if channel_location == "first":
        mean_vals = np.mean(array_float, axis=(2, 3), keepdims=True)
        std_vals = np.std(array_float, axis=(2, 3), keepdims=True)
    else:  # channel_location is None or "last"
        mean_vals = np.mean(array_float, axis=(1, 2), keepdims=True)
        std_vals = np.std(array_float, axis=(1, 2), keepdims=True)
    
    # Normalize: (x - mean) / std, handle std=0 case
    return np.where(
        std_vals > 0,
        (array_float - mean_vals) / std_vals,
        0.0
    )
