"""Tests for `fenn.parser.Parser.load_configuration`."""

import pytest

from fenn.parser import Parser


@pytest.fixture(autouse=True)
def _reset_parser():
    Parser.reset()
    yield
    Parser.reset()


def test_load_configuration_reads_a_mapping(tmp_path):
    cfg = tmp_path / "fenn.yaml"
    cfg.write_text("logger:\n  dir: logs\n")

    args = Parser(cfg).load_configuration()

    assert args["logger"] == {"dir": "logs"}
    assert args["project"] == "fenn"


def test_load_configuration_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        Parser(tmp_path / "missing.yaml").load_configuration()


def test_load_configuration_empty_file_raises_value_error(tmp_path):
    cfg = tmp_path / "fenn.yaml"
    cfg.write_text("")

    with pytest.raises(ValueError, match="empty"):
        Parser(cfg).load_configuration()


def test_load_configuration_non_mapping_raises_value_error(tmp_path):
    cfg = tmp_path / "fenn.yaml"
    cfg.write_text("- just\n- a\n- list\n")

    with pytest.raises(ValueError, match="mapping"):
        Parser(cfg).load_configuration()
