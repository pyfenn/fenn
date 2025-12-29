import logging
from typing import Any, Dict, List, Literal, TypedDict
import numpy as np

from fenn.vision.vision_utils import detect_format

logger = logging.getLogger(__name__)


class BatchIssue(TypedDict):
    """Details about an issue found in the batch."""
    severity: Literal["error", "warning", "info"]  # Severity level of the issue
    category: str  # Issue category (e.g., 'format', 'non_finite', 'outliers', 'dtype')
    message: str  # Human-readable description of the issue


class BatchReport(TypedDict):
    """Structured report of batch validation results."""
    is_valid: bool  # True if no errors found
    issues: List[BatchIssue]  # List of detected issues -> count
    summary: Dict[str, Any]  # Summary statistics about the batch


def check_image_batch(array) -> BatchReport:
    """
    Performs validation for mixed dtypes, unexpected channel counts, extreme outliers,
    and non-finite values, returning a structured report rather than raising immediately.
    
    This function does not modify the input array. It performs read-only validation
    checks to identify potential data quality issues that could cause problems during
    model training or inference.
    
    Args:
        array: Image array in batch format. The first dimension (N) must represent
            the batch size. Supported formats:
            - (N, H, W, C) - batch of images with channels last
            - (N, C, H, W) - batch of images with channels first
            - (N, H, W) - batch of grayscale images
        outlier_threshold: Number of standard deviations beyond which values are
            considered outliers. Default is 5.0 sigma.
    
    Returns:
        Dictionary containing:
            - is_valid: True if no errors found (warnings are acceptable)
            - issues: List of detected issues with severity, category, and message
            - summary: Summary statistics (batch_size, shape, dtype, channels, etc.)
    """
    issues: List[BatchIssue] = []
    outlier_threshold = 5.0
    try:
        format_info = detect_format(array)
        batch_size = array.shape[0]
        is_grayscale = format_info["is_grayscale"]
        channel_location = format_info["channel_location"]

        # Extract channel count
        if channel_location == "last":
            actual_channels = array.shape[3]
        elif channel_location == "first":
            actual_channels = array.shape[1]
        else:
            actual_channels = 1

    except (ValueError, TypeError) as e:
        issues.append(
            {
                "severity": "error",
                "category": "format",
                "message": f"Invalid array format: {str(e)}",
            }
        )
        return {
            "is_valid": False,
            "issues": issues,
            "summary": {"checks_completed": False},
        }

    # Check for non-finite values (NaN, Inf)
    has_nan = bool(np.isnan(array).any())
    has_inf = bool(np.isinf(array).any())

    if has_nan:
        nan_count = int(np.isnan(array).sum())
        nan_pct = (nan_count / array.size) * 100
        issues.append(
            {
                "severity": "error",
                "category": "non_finite",
                "message": f"Found {nan_count} NaN values ({nan_pct:.2f}% of pixels)",
            }
        )

    if has_inf:
        inf_count = int(np.isinf(array).sum())
        inf_pct = (inf_count / array.size) * 100
        issues.append(
            {
                "severity": "error",
                "category": "non_finite",
                "message": f"Found {inf_count} Inf values ({inf_pct:.2f}% of pixels)",
            }
        )

    # Check for empty batch first (before any calculations)
    if batch_size == 0:
        issues.append(
            {
                "severity": "error",
                "category": "batch_size",
                "message": "Batch is empty (batch_size=0)",
            }
        )

    # Check value range (only for non-empty arrays)
    if batch_size > 0 and not (
        has_nan and has_inf
    ):  # Only if we have valid values to check
        actual_min = float(np.nanmin(array))
        actual_max = float(np.nanmax(array))

        # Detect extreme outliers using z-score
        if array.size > 1:  # Need at least 2 values for std
            mean = float(np.nanmean(array))
            std = float(np.nanstd(array))

            if std > 0:  # Avoid division by zero
                # Use a sample for large arrays to speed up outlier detection
                sample_size = min(10000, array.size)
                if array.size > sample_size:
                    sample_indices = np.random.choice(
                        array.size, sample_size, replace=False
                    )
                    flat_array = array.ravel()
                    sample = flat_array[sample_indices]
                else:
                    sample = array.ravel()

                z_scores = np.abs((sample - mean) / std)
                outlier_mask = z_scores > outlier_threshold
                outlier_count = int(np.sum(outlier_mask))

                if outlier_count > 0:
                    outlier_pct = (outlier_count / len(sample)) * 100

                    # Consider percentage and absolute threshold
                    severity = "warning" if outlier_pct < 1.0 else "error"
                    issues.append(
                        {
                            "severity": severity,
                            "category": "outliers",
                            "message": f"Found {outlier_count} extreme outliers (>{outlier_threshold} sigma, {outlier_pct:.2f}% of sample)",
                        }
                    )

    # Check for unusual dtypes for images
    if array.dtype.kind not in ["u", "i", "f"]:  # unsigned int, int, float
        issues.append(
            {
                "severity": "warning",
                "category": "dtype",
                "message": f"Unusual dtype for images: {array.dtype} (kind={array.dtype.kind})",
            }
        )

    summary = {
        "batch_size": batch_size,
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "channels": actual_channels,
        "is_grayscale": is_grayscale,
        "total_issues": len(issues),
        "error_count": sum(1 for issue in issues if issue["severity"] == "error"),
        "warning_count": sum(1 for issue in issues if issue["severity"] == "warning"),
        "info_count": sum(1 for issue in issues if issue["severity"] == "info"),
        "checks_completed": True,
    }

    is_valid = summary["error_count"] == 0

    if issues:
        for issue in issues:
            if issue["severity"] == "error":
                logger.error(f"{issue['category']}: {issue['message']}")
            elif issue["severity"] == "warning":
                logger.warning(f"{issue['category']}: {issue['message']}")
            else:
                logger.info(f"{issue['category']}: {issue['message']}")

    return {"is_valid": is_valid, "issues": issues, "summary": summary}
