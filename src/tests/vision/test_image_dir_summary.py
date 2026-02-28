import pytest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from fenn.experimental.vision import image_dir_summary


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory with test images."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create various test images
    images = [
        ("test1.jpg", (400, 300), "JPEG", "RGB"),
        ("test2.jpg", (400, 300), "JPEG", "RGB"),
        ("test3.jpg", (400, 300), "JPEG", "RGB"),
        ("test4.png", (800, 600), "PNG", "RGB"),
        ("test5.jpg", (200, 200), "JPEG", "L"),
        ("test6.jpg", (1920, 1080), "JPEG", "RGB"),
    ]
    
    for name, size, format, mode in images:
        img = Image.new(mode, size, color="white")
        img.save(temp_dir / name, format=format)
    
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_image_dir_with_bad_files():
    """Create a temporary directory with test images and bad files."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Valid images
    img = Image.new("RGB", (100, 100), color="white")
    img.save(temp_dir / "good.jpg", "JPEG")
    
    # Bad files
    (temp_dir / "bad_empty.jpg").write_bytes(b"")
    (temp_dir / "bad_text.jpg").write_text("not an image")
    
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_image_dir_with_subdir():
    """Create a temporary directory with subdirectory."""
    temp_dir = Path(tempfile.mkdtemp())
    
    img = Image.new("RGB", (100, 100), color="white")
    img.save(temp_dir / "root.jpg", "JPEG")
    
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    img.save(subdir / "sub.jpg", "JPEG")
    
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestImageDirSummary:
    """Test suite for image_dir_summary function."""

    def test_basic_summary(self, temp_image_dir):
        """Test basic summary functionality."""
        result = image_dir_summary(temp_image_dir, recursive=False)
        
        assert result["total_count"] == 6
        assert "jpeg" in result["formats"]
        assert "png" in result["formats"]
        assert result["formats"]["jpeg"] == 5
        assert result["formats"]["png"] == 1
        assert "RGB" in result["color_modes"]
        assert "GRAY" in result["color_modes"]
        assert len(result["examples"]) > 0
        assert len(result["resolutions"]) > 0

    def test_recursive_search(self, temp_image_dir_with_subdir):
        """Test recursive directory search."""
        result_nonrec = image_dir_summary(temp_image_dir_with_subdir, recursive=False)
        result_rec = image_dir_summary(temp_image_dir_with_subdir, recursive=True)
        
        assert result_nonrec["total_count"] == 1
        assert result_rec["total_count"] == 2

    def test_failed_files_tracking(self, temp_image_dir_with_bad_files):
        """Test tracking of files that fail to read."""
        result = image_dir_summary(temp_image_dir_with_bad_files, recursive=False)
        
        assert result["total_count"] == 1
        assert result["failed_count"] == 2
        assert len(result["failed_files"]) == 2
        assert any("bad_empty" in f for f in result["failed_files"])
        assert any("bad_text" in f for f in result["failed_files"])

    def test_examples_diversity(self, temp_image_dir):
        """Test that examples show diversity in formats and modes."""
        result = image_dir_summary(temp_image_dir, recursive=False, max_examples=10)
        
        formats_in_examples = {e["format"].lower() for e in result["examples"]}
        modes_in_examples = {e["mode"] for e in result["examples"]}
        
        assert "jpeg" in formats_in_examples or "png" in formats_in_examples
        assert "RGB" in modes_in_examples or "GRAY" in modes_in_examples

    def test_max_examples_limit(self, temp_image_dir):
        """Test that max_examples limits the number of examples."""
        result = image_dir_summary(temp_image_dir, recursive=False, max_examples=3)
        
        assert len(result["examples"]) <= 3

    def test_nonexistent_directory(self):
        """Test error handling for nonexistent directory."""
        with pytest.raises(ValueError, match="Path does not exist"):
            image_dir_summary("/nonexistent/path", recursive=False)

    def test_file_not_directory(self, temp_image_dir):
        """Test error handling when path is a file, not directory."""
        test_file = temp_image_dir / "test1.jpg"
        with pytest.raises(ValueError, match="Path is not a directory"):
            image_dir_summary(test_file, recursive=False)

    def test_empty_directory(self):
        """Test handling of empty directory."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            result = image_dir_summary(temp_dir, recursive=False)
            assert result["total_count"] == 0
            assert len(result["examples"]) == 0
            assert result["failed_count"] == 0
        finally:
            shutil.rmtree(temp_dir)

