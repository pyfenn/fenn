import pytest
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch, Mock
import requests

from fenn.cli.pull_command import (
    execute,
    _download_template,
    TemplateNotFoundError,
    NetworkError,
    TemplateError,
)


class TestPullCommand:
    """Test suite for the fenn pull command."""

    def test_pull_success(self, tmp_path, requests_mock):
        """Test successful template pull."""
        args = Mock()
        args.template = "base"
        args.path = str(tmp_path)
        args.force = False

        # Mock GitHub API response for template existence check
        requests_mock.get(
            "https://api.github.com/repos/pyfenn/templates/contents/base",
            status_code=200,
            json={"name": "base", "type": "dir"}
        )

        # Create a mock zip file content
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, 'w') as zf:
                zf.writestr("templates-main/base/main.py", "print('hello')")
                zf.writestr("templates-main/base/fenn.yaml", "project: test")
                zf.writestr("templates-main/base/.gitignore", ".env")
            
            # Mock archive download
            with open(tmp_zip.name, 'rb') as f:
                zip_content = f.read()
            
            requests_mock.get(
                "https://github.com/pyfenn/templates/archive/refs/heads/main.zip",
                status_code=200,
                content=zip_content
            )

            execute(args)
        
        # Verify files were extracted
        assert (tmp_path / "main.py").exists()
        assert (tmp_path / "fenn.yaml").exists()
        assert (tmp_path / ".gitignore").exists()
        assert (tmp_path / "main.py").read_text() == "print('hello')"

    def test_pull_template_not_found(self, requests_mock, capsys, tmp_path):
        """Test pull with non-existent template."""
        args = Mock()
        args.template = "nonexistent"
        args.path = str(tmp_path)
        args.force = False

        requests_mock.get(
            "https://api.github.com/repos/pyfenn/templates/contents/nonexistent",
            status_code=404
        )

        with pytest.raises(SystemExit) as exc_info:
            execute(args)
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Template" in captured.out
        assert "nonexistent" in captured.out

    def test_pull_network_error_on_check(self, requests_mock, capsys, tmp_path):
        """Test pull with network error during template check."""
        args = Mock()
        args.template = "base"
        args.path = str(tmp_path)
        args.force = False

        requests_mock.get(
            "https://api.github.com/repos/pyfenn/templates/contents/base",
            exc=requests.exceptions.ConnectionError("Network error")
        )

        with pytest.raises(SystemExit) as exc_info:
            execute(args)
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Network error" in captured.out

    def test_pull_network_error_on_download(self, requests_mock, capsys, tmp_path):
        """Test pull with network error during archive download."""
        args = Mock()
        args.template = "base"
        args.path = str(tmp_path)
        args.force = False

        requests_mock.get(
            "https://api.github.com/repos/pyfenn/templates/contents/base",
            status_code=200,
            json={"name": "base", "type": "dir"}
        )

        requests_mock.get(
            "https://github.com/pyfenn/templates/archive/refs/heads/main.zip",
            exc=requests.exceptions.ConnectionError("Network error")
        )

        with pytest.raises(SystemExit) as exc_info:
            execute(args)
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Network error" in captured.out

    def test_pull_non_empty_directory_without_force(self, tmp_path, capsys):
        """Test pull into non-empty directory without --force."""
        # Create a file in the directory
        (tmp_path / "existing.txt").write_text("existing")

        args = Mock()
        args.template = "base"
        args.path = str(tmp_path)
        args.force = False

        with pytest.raises(SystemExit) as exc_info:
            execute(args)
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "non-empty directory" in captured.out
        assert "--force" in captured.out

    def test_pull_non_empty_directory_with_force(self, tmp_path, requests_mock):
        """Test pull into non-empty directory with --force."""
        (tmp_path / "existing.txt").write_text("existing")

        args = Mock()
        args.template = "base"
        args.path = str(tmp_path)
        args.force = True

        requests_mock.get(
            "https://api.github.com/repos/pyfenn/templates/contents/base",
            status_code=200,
            json={"name": "base", "type": "dir"}
        )

        # Create a mock zip file content
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, 'w') as zf:
                zf.writestr("templates-main/base/main.py", "print('hello')")
            
            with open(tmp_zip.name, 'rb') as f:
                zip_content = f.read()
            
            requests_mock.get(
                "https://github.com/pyfenn/templates/archive/refs/heads/main.zip",
                status_code=200,
                content=zip_content
            )

            execute(args)

        # Verify new files were extracted
        assert (tmp_path / "main.py").exists()

    def test_pull_missing_template_name(self, capsys):
        """Test pull without template name."""
        args = Mock()
        args.template = ""
        args.path = "."
        args.force = False

        with pytest.raises(SystemExit) as exc_info:
            execute(args)
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Template name is required" in captured.out

    def test_download_template_empty_template(self, requests_mock, tmp_path):
        """Test downloading an empty template."""
        requests_mock.get(
            "https://api.github.com/repos/pyfenn/templates/contents/base",
            status_code=200,
            json={"name": "base", "type": "dir"}
        )

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, 'w') as zf:
                # Create zip with no template files (only other files)
                zf.writestr("templates-main/other/file.txt", "content")
            
            with open(tmp_zip.name, 'rb') as f:
                zip_content = f.read()
            
            requests_mock.get(
                "https://github.com/pyfenn/templates/archive/refs/heads/main.zip",
                status_code=200,
                content=zip_content
            )

            with pytest.raises(TemplateError) as exc_info:
                _download_template("base", tmp_path, False)
            
            assert "base" in str(exc_info.value)
            assert "empty" in str(exc_info.value).lower()
            assert "unexpected structure" in str(exc_info.value).lower()

    def test_download_template_nested_structure(self, requests_mock, tmp_path):
        """Test downloading template with nested directory structure."""
        requests_mock.get(
            "https://api.github.com/repos/pyfenn/templates/contents/base",
            status_code=200,
            json={"name": "base", "type": "dir"}
        )

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, 'w') as zf:
                zf.writestr("templates-main/base/main.py", "print('hello')")
                zf.writestr("templates-main/base/models/model.py", "class Model: pass")
                zf.writestr("templates-main/base/dataset/data.py", "data = []")
            
            with open(tmp_zip.name, 'rb') as f:
                zip_content = f.read()
            
            requests_mock.get(
                "https://github.com/pyfenn/templates/archive/refs/heads/main.zip",
                status_code=200,
                content=zip_content
            )

            _download_template("base", tmp_path, False)

        # Verify nested structure was preserved
        assert (tmp_path / "main.py").exists()
        assert (tmp_path / "models" / "model.py").exists()
        assert (tmp_path / "dataset" / "data.py").exists()

    def test_download_template_http_error_500(self, requests_mock):
        """Test handling of HTTP 500 error."""
        requests_mock.get(
            "https://api.github.com/repos/pyfenn/templates/contents/base",
            status_code=500
        )

        with pytest.raises(NetworkError) as exc_info:
            _download_template("base", Path(tempfile.mkdtemp()), False)
        
        assert "Failed to check template existence" in str(exc_info.value)
        assert "500" in str(exc_info.value)

    def test_download_template_http_error_403(self, requests_mock):
        """Test handling of HTTP 403 error (rate limit, etc.)."""
        requests_mock.get(
            "https://api.github.com/repos/pyfenn/templates/contents/base",
            status_code=403
        )

        with pytest.raises(NetworkError) as exc_info:
            _download_template("base", Path(tempfile.mkdtemp()), False)
        
        assert "Failed to check template existence" in str(exc_info.value)
        assert "403" in str(exc_info.value)

    def test_download_template_empty_directories(self, requests_mock, tmp_path):
        """Test downloading template with empty directories."""
        requests_mock.get(
            "https://api.github.com/repos/pyfenn/templates/contents/base",
            status_code=200,
            json={"name": "base", "type": "dir"}
        )

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, 'w') as zf:
                zf.writestr("templates-main/base/main.py", "print('hello')")
                zf.writestr("templates-main/base/logger/", "")  # Empty directory
                zf.writestr("templates-main/base/dataset/", "")  # Empty directory
                zf.writestr("templates-main/base/models/", "")  # Empty directory
            
            with open(tmp_zip.name, 'rb') as f:
                zip_content = f.read()
            
            requests_mock.get(
                "https://github.com/pyfenn/templates/archive/refs/heads/main.zip",
                status_code=200,
                content=zip_content
            )

            _download_template("base", tmp_path, False)

        # Verify files and empty directories were created
        assert (tmp_path / "main.py").exists()
        assert (tmp_path / "logger").is_dir()
        assert (tmp_path / "dataset").is_dir()
        assert (tmp_path / "models").is_dir()
        # Verify directories are empty
        assert not any((tmp_path / "logger").iterdir())
        assert not any((tmp_path / "dataset").iterdir())
        assert not any((tmp_path / "models").iterdir())
