import numpy as np
from typing import Tuple, Union

try:
    import torch
    from torchvision.transforms import functional as F
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

from fenn.vision.vision_utils import detect_format


def resize_batch(
    array: np.ndarray,
    size: Union[int, Tuple[int, int]],
    interpolation: str = "bilinear"
) -> np.ndarray:
    """
    Resize a batch of images to a target size, preserving channel order and dtype where possible.

    Convenience wrapper to resize a batch of images to a uniform target size. The function
    detects the input format (channels first/last, grayscale/color) and preserves
    the same format in the output. Dtype is preserved where possible.

    Args:
        array: Input image array with batch dimension. Must be 3D or 4D:
            - (N, H, W) - batch of grayscale images
            - (N, H, W, C) - batch with channels last
            - (N, C, H, W) - batch with channels first

        size: Target size for resizing. Can be:
            - int: Single value for both height and width (square output)
            - Tuple[int, int]: (height, width) for rectangular output

        interpolation: Interpolation method to use. Default is "bilinear".
            Supported methods: "nearest", "nearest_exact", "bilinear", "bicubic"

    Returns:
        Resized image array with the same:
            - Batch size (N)
            - Channel order (first/last/none)
            - Dtype (where possible)
            - Shape: (N, size[0], size[1]) for grayscale or
                     (N, size[0], size[1], C) / (N, C, size[0], size[1]) for color

    Raises:
        TypeError: If array is not a numpy array
        ValueError: If array doesn't have batch dimension, has unsupported format,
                    or size is invalid
        ImportError: If torchvision is not installed (required for resizing)

    Example:
        >>> import numpy as np
        >>> from fenn.vision import resize_batch
        >>> # Batch of color images with channels last
        >>> images = np.random.randint(0, 255, (32, 224, 224, 3), dtype=np.uint8)
        >>> resized = resize_batch(images, (128, 128))
        >>> print(resized.shape)  # (32, 128, 128, 3)
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(array)}")

    # Validate size parameter
    if isinstance(size, int):
        target_height, target_width = size, size
    elif isinstance(size, tuple) and len(size) == 2:
        target_height, target_width = size
    else:
        raise ValueError(
            f"size must be an int or tuple of (height, width), got {type(size)}"
        )

    if target_height <= 0 or target_width <= 0:
        raise ValueError(
            f"size dimensions must be positive, got ({target_height}, {target_width})"
        )

    # Detect format to preserve channel order
    format_info = detect_format(array)
    channel_location = format_info["channel_location"]

    # Validate interpolation method
    valid_interpolations = ["nearest", "nearest_exact", "bilinear", "bicubic"]
    if interpolation not in valid_interpolations:
        raise ValueError(
            f"interpolation must be one of {valid_interpolations}, got '{interpolation}'"
        )

    if not TORCHVISION_AVAILABLE:
        raise ImportError(
            "torchvision is required for resize_batch. "
            "Install it with: pip install fenn[torch] or pip install torchvision"
        )

    interpolation_map = {
        "nearest": F.InterpolationMode.NEAREST,
        "nearest_exact": F.InterpolationMode.NEAREST_EXACT,
        "bilinear": F.InterpolationMode.BILINEAR,
        "bicubic": F.InterpolationMode.BICUBIC,
    }
    torch_interpolation = interpolation_map[interpolation]

    # Store original dtype and shape info
    original_dtype = array.dtype
    batch_size = array.shape[0]

    # Get determine whether we are downsampling
    if channel_location == "last":
        # (N, H, W, C)
        original_height, original_width = array.shape[1], array.shape[2]
    elif channel_location == "first":
        # (N, C, H, W)
        original_height, original_width = array.shape[2], array.shape[3]
    else:
        # (N, H, W)
        original_height, original_width = array.shape[1], array.shape[2]
    is_downsampling = (original_height > target_height) or (original_width > target_width)

    # Normalize to channels-first format (N, C, H, W) for torchvision
    if channel_location == "last":
        # (N, H, W, C) -> (N, C, H, W)
        array = np.transpose(array, (0, 3, 1, 2))
    elif channel_location is None:
        # (N, H, W) -> (N, 1, H, W) for grayscale
        array = array[:, np.newaxis, :, :]

    # Convert to torch tensor
    if original_dtype == np.uint8:
        # Use uint8 directly - torchvision supports it natively
        tensor = torch.from_numpy(array.astype(np.uint8))
        needs_normalization = False
    elif np.issubdtype(original_dtype, np.floating):
        # Float arrays: assume [0, 1] range, convert to float32
        tensor = torch.from_numpy(array.astype(np.float32))
        needs_normalization = False
    else:
        # Other integer types: convert to float32 and normalize to [0, 1]
        tensor = torch.from_numpy(array.astype(np.float32))
        max_val = float(np.iinfo(original_dtype).max)
        tensor = tensor / max_val
        needs_normalization = True

    # Enable antialiasing only when downsampling with smooth interpolation methods
    use_antialias = (
        is_downsampling and
        interpolation in ["bilinear", "bicubic"]
    )

    # Resize using torchvision
    resized_tensor = F.resize(
        tensor,
        size=[target_height, target_width],
        interpolation=torch_interpolation,
        antialias=use_antialias
    )

    # Convert back to numpy
    result = resized_tensor.numpy()

    # Convert back to original dtype
    if needs_normalization:
        max_val = float(np.iinfo(original_dtype).max)
        result = (result * max_val).clip(0, max_val).astype(original_dtype)
    else:
        # uint8 or float - just ensure correct dtype
        result = result.astype(original_dtype)
        if np.issubdtype(original_dtype, np.floating):
            result = result.clip(0, 1)

    # Restore original channel order
    if channel_location == "last":
        # (N, C, H, W) -> (N, H, W, C)
        result = np.transpose(result, (0, 2, 3, 1))
    elif channel_location is None:
        # (N, 1, H, W) -> (N, H, W)
        result = result.squeeze(1)

    return result
