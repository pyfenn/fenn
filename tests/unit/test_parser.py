import pytest

from fenn.parser import Parser


@pytest.fixture(autouse=True)
def _reset_parser():
    Parser.reset()
    yield
    Parser.reset()


def _load(tmp_path, content):
    path = tmp_path / "cfg.yaml"
    path.write_text(content)
    return Parser(path).load_configuration()


def test_empty_file_raises(tmp_path):
    with pytest.raises(ValueError, match="cfg.yaml"):
        _load(tmp_path, "")


def test_list_top_level_raises(tmp_path):
    with pytest.raises(ValueError, match="list"):
        _load(tmp_path, "- a\n- b\n")


def test_scalar_top_level_raises(tmp_path):
    with pytest.raises(ValueError, match="str"):
        _load(tmp_path, "just a string\n")


def test_valid_yaml_sets_project(tmp_path):
    args = _load(tmp_path, "lr: 0.1\n")
    assert args == {"lr": 0.1, "project": "cfg"}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Parser(tmp_path / "nope.yaml").load_configuration()


def test_failed_load_leaves_args_untouched(tmp_path):
    with pytest.raises(ValueError):
        _load(tmp_path, "- a\n")
    assert Parser().args == {}
    with pytest.raises(RuntimeError):
        Parser().print()
