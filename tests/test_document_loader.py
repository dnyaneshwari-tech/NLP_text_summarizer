from pathlib import Path

import pytest

from src.document_loader import extract_document_text, validate_upload_file


def test_txt_extraction(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("Alpha beta gamma.\nThis is a second line.\n", encoding="utf-8")

    result = extract_document_text(str(sample), "sample.txt")

    assert "Alpha beta gamma" in result
    assert "This is a second line" in result


def test_empty_txt_input(tmp_path):
    sample = tmp_path / "empty.txt"
    sample.write_text("", encoding="utf-8")

    result = extract_document_text(str(sample), "empty.txt")

    assert result == ""


def test_unsupported_extension(tmp_path):
    sample = tmp_path / "sample.md"
    sample.write_text("Not supported", encoding="utf-8")

    with pytest.raises(ValueError):
        extract_document_text(str(sample), "sample.md")


def test_basic_extraction_validation(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("Basic text content", encoding="utf-8")

    assert validate_upload_file(str(sample), "sample.txt") is True
    assert validate_upload_file(str(sample), "sample.pdf") is False
