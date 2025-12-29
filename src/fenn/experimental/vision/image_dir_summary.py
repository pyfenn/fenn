from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional
from collections import Counter
import logging

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from fenn.vision.vision_utils import normalize_color_mode

logger = logging.getLogger(__name__)


class ImageDirSummary(TypedDict):
    """Summary information for a directory of images."""
    total_count: int
    formats: Dict[str, int]  # Format extension -> count
    resolutions: Dict[str, int]  # Resolution string like "1920x1080" -> count
    color_modes: Dict[str, int]  # Color mode -> count (e.g., 'RGB', 'GRAY', 'RGBA')
    examples: List[Dict[str, Any]]  # Sample file paths with metadata
    failed_files: List[str]  # Paths of files that couldn't be read
    failed_count: int  # Number of files that couldn't be read


def _select_examples(
    all_metadata: List[Dict[str, Any]],
    resolution_counter: Counter,
    format_counter: Counter,
    color_mode_counter: Counter,
    total_count: int,
    max_examples: int
) -> List[Dict[str, Any]]:
    """
    Select representative examples that show diversity in resolutions, formats, and modes.
    
    Prioritizes format and color mode diversity (most critical), then resolution diversity.
    """
    if not all_metadata or max_examples <= 0:
        return []
    
    threshold = total_count * 0.1
    common_res_set = {
        res for res, count in resolution_counter.most_common()
        if count >= threshold
    }
    common_format_set = {
        fmt for fmt, count in format_counter.most_common()
        if count >= threshold
    }
    common_mode_set = {
        mode for mode, count in color_mode_counter.most_common()
        if count >= threshold
    }
    
    selected = []
    used_paths = set()
    used_resolutions = set()
    used_formats = set()
    used_modes = set()
    remaining = [m for m in all_metadata]
    
    while len(selected) < max_examples and remaining:
        best_candidate = None
        best_score = -1
        
        for meta in remaining:
            if meta["path"] in used_paths:
                continue
            
            score = 0.0
            
            if meta["format_lower"] not in used_formats:
                score += 20.0
            elif meta["format_lower"] in common_format_set:
                score += 3.0
            
            if meta["mode"] not in used_modes:
                score += 18.0
            elif meta["mode"] in common_mode_set:
                score += 3.0
            
            if meta["resolution"] not in used_resolutions:
                score += 10.0
            if meta["resolution"] in common_res_set:
                score += 5.0
        
            if score > best_score:
                best_score = score
                best_candidate = meta
        
        if best_candidate is None:
            # No more valid candidates
            break
        
        # Add best candidate
        selected.append({
            "path": best_candidate["path"],
            "width": best_candidate["width"],
            "height": best_candidate["height"],
            "format": best_candidate["format"],
            "mode": best_candidate["mode"]
        })
        used_paths.add(best_candidate["path"])
        used_resolutions.add(best_candidate["resolution"])
        used_formats.add(best_candidate["format_lower"])
        used_modes.add(best_candidate["mode"])
        remaining.remove(best_candidate)
    
    return selected


def image_dir_summary(
    path: str | Path,
    recursive: bool = False,
    max_examples: int = 5
) -> ImageDirSummary:
    """
    Scan a folder (optionally recursive) and summarize counts, common resolutions,
    formats, and a few representative examples.
    
    This function does not load all pixels into memory at once. Instead, it reads
    only image metadata (dimensions, format) to keep memory usage low.
    
    Args:
        path: Path to the directory to scan. Can be a string or Path object.
        recursive: If True, scan subdirectories recursively. Default is False.
        max_examples: Maximum number of representative examples to include.
            Default is 5.
    
    Returns:
        Dictionary containing:
            - total_count: Total number of images found
            - formats: Dictionary mapping format extensions to counts
            - resolutions: Dictionary mapping resolution strings (e.g., "1920x1080") to counts
            - color_modes: Dictionary mapping color modes (e.g., "RGB", "L", "RGBA") to counts
            - examples: List of sample file paths with their metadata
            - failed_files: List of file paths that couldn't be read
            - failed_count: Number of files that couldn't be read
    
    Raises:
        ImportError: If PIL/Pillow is not installed (required for reading image metadata)
        ValueError: If path does not exist or is not a directory
        OSError: If there are permission issues accessing the directory    
    """
    if not PIL_AVAILABLE:
        raise ImportError(
            "PIL/Pillow is required for image_dir_summary. "
            "Install it with: pip install fenn[vision] or pip install Pillow"
        )
    
    path = Path(path)
    
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    
    # Statistics to collect
    total_count = 0
    format_counter = Counter()
    resolution_counter = Counter()
    color_mode_counter = Counter()
    all_image_metadata = []  # Store all metadata for representative selection
    failed_files = []  # Track files that couldn't be read
    
    # Pattern for finding files
    if recursive:
        image_files = [
            f for f in path.rglob('*')
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    else:
        image_files = [
            f for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    
    # Scan files and collect metadata
    for img_path in image_files:
        try:
            # Open image to read metadata without loading pixels
            with Image.open(img_path) as img:
                width, height = img.size
                format_name = img.format or img_path.suffix[1:].lower()
                resolution_str = f"{width}x{height}"
                normalized_mode = normalize_color_mode(img.mode)
                
                total_count += 1
                format_counter[format_name.lower()] += 1
                resolution_counter[resolution_str] += 1
                color_mode_counter[normalized_mode] += 1
                
                all_image_metadata.append({
                    "path": str(img_path),
                    "width": width,
                    "height": height,
                    "format": format_name,
                    "mode": normalized_mode,
                    "resolution": resolution_str,
                    "format_lower": format_name.lower()
                })
        
        except Exception as e:
            failed_files.append(str(img_path))
            logger.warning(f"Failed to read image {img_path}: {e}")
            continue
    
    # Select examples that show diversity
    examples = _select_examples(
        all_image_metadata, 
        resolution_counter,
        format_counter,
        color_mode_counter,
        total_count,
        max_examples
    )
    
    return {
        "total_count": total_count,
        "formats": dict(format_counter),
        "resolutions": dict(resolution_counter),
        "color_modes": dict(color_mode_counter),
        "examples": examples,
        "failed_files": failed_files,
        "failed_count": len(failed_files)
    }

