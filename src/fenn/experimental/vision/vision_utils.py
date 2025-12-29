import numpy as np
from typing import Dict, Literal, TypedDict


class FormatInfo(TypedDict):
    """Format information for an image batch array."""
    is_grayscale: bool
    channel_location: Literal["first", "last"] | None


def detect_format(array: np.ndarray) -> FormatInfo:
    """
    Detect the format of an image batch array.
    
    The array must have a batch dimension as the first dimension (N).
    
    Args:
        array: Input image array with batch dimension. Must be 3D or 4D:
            - (N, H, W) - batch of grayscale images
            - (N, H, W, C) - batch with channels last
            - (N, C, H, W) - batch with channels first
        
    Returns:
        Dictionary with format information:
            - is_grayscale: bool - True if grayscale (C=1 or no channel dimension)
            - channel_location: Literal["first", "last"] | None - Channel position.
              "first" for (N, C, H, W), "last" for (N, H, W, C), None for (N, H, W) with no channel dimension
    
    Raises:
        TypeError: If array is not a numpy array
        ValueError: If array doesn't have batch dimension or has unsupported format
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(array)}")
    
    shape = array.shape
    ndim = array.ndim
    
    # Must have batch dimension - reject 2D arrays
    if ndim < 3:
        raise ValueError(
            f"Array must have batch dimension. Expected 3D greyscale (N, H, W) or 4D color "
            f"(N, H, W, C) / (N, C, H, W), got {ndim}D array with shape {shape}. "
            f"For single images, wrap with array[np.newaxis, ...] to add batch dimension."
        )
    
    # Handle 3D arrays: (N, H, W) - grayscale batch
    if ndim == 3:
        return {"is_grayscale": True, "channel_location": None}
    
    # Handle 4D arrays: (N, H, W, C) or (N, C, H, W)
    elif ndim == 4:
        # Explicitly detect grayscale: C=1
        if shape[1] == 1:
            # (N, 1, H, W) - grayscale with channels first
            return {"is_grayscale": True, "channel_location": "first"}
        elif shape[3] == 1:
            # (N, H, W, 1) - grayscale with channels last
            return {"is_grayscale": True, "channel_location": "last"}
        # Check if last dimension is small (channels) - channels last
        elif shape[3] <= 4:
            # (N, H, W, C) where C > 1
            return {"is_grayscale": False, "channel_location": "last"}
        # Check if second dimension is small (channels) - channels first
        elif shape[1] <= 4:
            # (N, C, H, W) where C > 1
            return {"is_grayscale": False, "channel_location": "first"}
        else:
            # Invalid format - neither dimension looks like channels
            raise ValueError(
                f"Unable to detect format for 4D array with shape {shape}. "
                f"Expected (N, H, W, C) or (N, C, H, W) where C is 1-4."
            )
    
    else:
        raise ValueError(
            f"Unsupported array dimension: {ndim}. "
            f"Expected 3D (N, H, W) or 4D (N, H, W, C) / (N, C, H, W), got shape {shape}"
        )


def normalize_color_mode(mode: str) -> str:
    """
    Normalize color mode names.
    
    Args:
        mode: Color mode string
    
    Returns:
        Normalized color mode string in uppercase (e.g., 'RGB', 'RGBA', 'GRAY')
    
    Raises:
        ValueError: If the mode is not supported. Supported modes are:
            'RGB', 'RGBA', 'GRAY', 'L' (case-insensitive)
    """
    mode_upper = mode.upper()
    
    supported_modes = {"RGB", "RGBA", "GRAY", "L"}
    
    if mode_upper not in supported_modes:
        raise ValueError(
            f"Unsupported color mode '{mode}'. "
            f"Supported modes: {', '.join(sorted(supported_modes))}"
        )

    if mode_upper == 'L':
        return 'GRAY'
    
    return mode_upper
