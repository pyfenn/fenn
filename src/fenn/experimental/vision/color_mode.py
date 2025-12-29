import numpy as np
from typing import Literal

from fenn.vision.vision_utils import detect_format, normalize_color_mode


def _gray_to_rgb(
    array: np.ndarray,
    channel_location: Literal["first", "last"] | None,
) -> np.ndarray:
    """
    Convert grayscale array to RGB by duplicating the channel.

    Args:
        array: Grayscale image array with batch dimension
        channel_location: Channel position ("first", "last", or None)

    Returns:
        RGB array with 3 channels
    """
    if channel_location is None:
        # (N, H, W) - no channel dimension → (N, H, W, 3) with channels last
        return np.stack([array, array, array], axis=-1)
    elif channel_location == "last":
        # (N, H, W, 1) - channels last → (N, H, W, 3)
        return np.repeat(array, 3, axis=-1)
    else:  # channel_location == "first"
        # (N, 1, H, W) - channels first → (N, 3, H, W)
        return np.repeat(array, 3, axis=1)


def _rgb_to_rgba(
    array: np.ndarray,
    channel_location: Literal["first", "last"],
) -> np.ndarray:
    """
    Convert RGB array to RGBA by adding alpha channel with full opacity.

    Args:
        array: RGB image array with batch dimension
        channel_location: Channel position ("first" or "last")

    Returns:
        RGBA array with 4 channels
    """
    # Get alpha value for dtype (full opacity)
    if array.dtype == np.uint8:
        alpha_value = 255
    elif array.dtype.kind == 'f':  # float
        alpha_value = 1.0
    else:
        alpha_value = np.iinfo(array.dtype).max

    if channel_location == "last":
        # (N, H, W, 3) → (N, H, W, 4)
        alpha = np.full((*array.shape[:-1], 1), alpha_value, dtype=array.dtype)
        return np.concatenate([array, alpha], axis=-1)
    else:  # channel_location == "first"
        # (N, 3, H, W) → (N, 4, H, W)
        alpha = np.full((array.shape[0], 1, *array.shape[2:]), alpha_value, dtype=array.dtype)
        return np.concatenate([array, alpha], axis=1)


def _rgba_to_rgb(
    array: np.ndarray,
    channel_location: Literal["first", "last"],
) -> np.ndarray:
    """
    Convert RGBA array to RGB by dropping alpha channel.

    Args:
        array: RGBA image array with batch dimension
        channel_location: Channel position ("first" or "last")

    Returns:
        RGB array with 3 channels
    """
    if channel_location == "last":
        # (N, H, W, 4) → (N, H, W, 3)
        return array[..., :3]
    else:  # channel_location == "first"
        # (N, 4, H, W) → (N, 3, H, W)
        return array[:, :3, ...]


def _rgb_to_gray(
    array: np.ndarray,
    channel_location: Literal["first", "last"],
) -> np.ndarray:
    """
    Convert RGB array to grayscale using standard luminance weights.

    Uses ITU-R BT.601 weights: 0.299*R + 0.587*G + 0.114*B

    Args:
        array: RGB image array with batch dimension
        channel_location: Channel position ("first" or "last")

    Returns:
        Grayscale array with 1 channel (or no channel dimension if channels were last)
    """
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)

    if channel_location == "last":
        # (N, H, W, 3) → (N, H, W)
        gray_float = np.sum(array.astype(np.float32) * weights, axis=-1)
        return gray_float.astype(array.dtype)
    else:  # channel_location == "first"
        # (N, 3, H, W) → (N, H, W)
        weights_reshaped = weights.reshape(3, 1, 1)
        gray_float = np.sum(array.astype(np.float32) * weights_reshaped, axis=1)
        return gray_float.astype(array.dtype)


def _convert_color_mode(
    array: np.ndarray,
    current_mode: str,
    target_mode: str,
    channel_location: Literal["first", "last"] | None,
) -> np.ndarray:
    """
    Convert array from current_mode to target_mode.

    Args:
        array: Input image array with batch dimension
        current_mode: Current color mode ("GRAY", "RGB", or "RGBA")
        target_mode: Target color mode ("GRAY", "RGB", or "RGBA")
        channel_location: Channel position ("first", "last", or None)

    Returns:
        Converted array in target mode
    """
    # Safety check: if already in target mode, return a copy
    if current_mode == target_mode:
        return array.copy()

    # Perform conversions using helper functions
    if current_mode == "GRAY":
        if target_mode == "RGB":
            return _gray_to_rgb(array, channel_location)
        else:  # target_mode == "RGBA"
            rgb = _gray_to_rgb(array, channel_location)
            effective_channel_location = channel_location if channel_location is not None else "last"
            return _rgb_to_rgba(rgb, effective_channel_location)

    elif current_mode == "RGB":
        if target_mode == "GRAY":
            return _rgb_to_gray(array, channel_location)
        else:  # target_mode == "RGBA"
            return _rgb_to_rgba(array, channel_location)

    else:  # current_mode == "RGBA"
        if target_mode == "RGB":
            return _rgba_to_rgb(array, channel_location)
        else:  # target_mode == "GRAY"
            rgb = _rgba_to_rgb(array, channel_location)
            return _rgb_to_gray(rgb, channel_location)


def ensure_color_mode(array: np.ndarray, mode: str = "RGB") -> np.ndarray:
    """
    Convert grayscale / RGB / RGBA images to the desired channel layout.
    
    Converts NumPy image arrays to the specified color mode by:
    - Expanding grayscale images to 3 channels (e.g., for RGB mode)
    - Dropping alpha channel (e.g., converting RGBA to RGB)
    - Preserving the original channel layout (first or last)

    Args:
        array: Input image array with batch dimension. Must be:
            - (N, H, W) - batch of grayscale images
            - (N, H, W, C) - batch with channels last
            - (N, C, H, W) - batch with channels first
        mode: Target color mode. Supported modes:
            - "RGB" - 3 channels (default)
            - "RGBA" - 4 channels with alpha
            - "L" or "GRAY" - 1 channel grayscale

    Returns:
        Array converted to the specified color mode, preserving:
            - Original dtype
            - Original channel layout (first or last)
            - Batch dimension

    Raises:
        TypeError: If array is not a numpy array
        ValueError: If mode is not supported or array format is invalid

    Example:
        >>> import numpy as np
        >>> # Convert grayscale to RGB
        >>> gray_batch = np.random.randint(0, 255, (10, 224, 224), dtype=np.uint8)
        >>> rgb = ensure_color_mode(gray_batch, mode="RGB")
        >>> print(rgb.shape)  # (10, 224, 224, 3)
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(array)}")

    # Validate and normalize target mode using utility function
    target_mode = normalize_color_mode(mode)

    format_info = detect_format(array)
    is_grayscale = format_info["is_grayscale"]
    channel_location: Literal["first", "last"] | None = format_info["channel_location"]

    # Determine current channel count and mode
    if is_grayscale:
        current_channels = 1
        current_mode = "GRAY"
    else:
        if channel_location == "last":
            current_channels = array.shape[3]
        elif channel_location == "first":
            current_channels = array.shape[1]
        else:
            # This should never happen - non-grayscale images must have channel_location
            raise ValueError(
                f"Invalid format: non-grayscale image with channel_location={channel_location}. "
                f"This indicates a bug in detect_format or unexpected array format."
            )

        if current_channels == 1:
            current_mode = "GRAY"
        elif current_channels == 3:
            current_mode = "RGB"
        elif current_channels == 4:
            current_mode = "RGBA"
        else:
            raise ValueError(
                f"Unsupported channel count: {current_channels}. "
                f"Expected 1 (grayscale), 3 (RGB), or 4 (RGBA)."
            )

    # Convert using helper function
    return _convert_color_mode(array, current_mode, target_mode, channel_location)
