from __future__ import annotations

import re

import nltk
from nltk.tokenize import sent_tokenize


def ensure_nltk_resources() -> None:
    """Ensure the sentence tokenizer resource is available."""
    try:
        sent_tokenize("test sentence.")
    except LookupError:
        nltk.download("punkt", quiet=True)
    except Exception:
        nltk.download("punkt", quiet=True)


ensure_nltk_resources()


def clean_whitespace(text: str) -> str:
    """Normalize repeated whitespace while preserving sentence punctuation."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n +", "\n", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """Standardize whitespace and blank lines without altering the content meaning."""
    cleaned = clean_whitespace(text)
    if not cleaned:
        return ""
    return cleaned


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK sentence tokenizer."""
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    try:
        sentences = sent_tokenize(cleaned)
    except Exception:
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [sentence.strip() for sentence in sentences if sentence and sentence.strip()]


def count_words(text: str) -> int:
    """Count words using a safe regex-based method."""
    if not text or not text.strip():
        return 0
    return len(re.findall(r"\b\w+\b", text))


def count_sentences(text: str) -> int:
    """Count sentences in the text."""
    return len(split_into_sentences(text))


def is_empty_text(text: str) -> bool:
    """Check whether the text is empty or nearly empty."""
    if text is None:
        return True
    cleaned = clean_whitespace(text)
    return cleaned == "" or len(cleaned) < 3
