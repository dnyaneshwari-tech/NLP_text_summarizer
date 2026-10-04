from __future__ import annotations

from typing import Any

from src.text_preprocessing import count_sentences, count_words


def calculate_compression_ratio(original_text: str, summary_text: str) -> float:
    """Return summary_word_count / original_word_count. Values are kept between 0 and 1."""
    original_count = count_words(original_text)
    if original_count == 0:
        return 0.0
    summary_count = count_words(summary_text)
    return round(summary_count / original_count, 4)


def calculate_percentage_reduction(original_text: str, summary_text: str) -> float:
    """Return the percentage reduction relative to the original length."""
    ratio = calculate_compression_ratio(original_text, summary_text)
    return round((1 - ratio) * 100, 2)


def generate_summary_stats(original_text: str, summary_text: str) -> dict[str, Any]:
    """Prepare a dictionary with summary statistics for display and download."""
    original_words = count_words(original_text)
    summary_words = count_words(summary_text)
    original_sentences = count_sentences(original_text)
    summary_sentences = count_sentences(summary_text)
    compression_ratio = calculate_compression_ratio(original_text, summary_text)
    reduction = calculate_percentage_reduction(original_text, summary_text)

    return {
        "original_word_count": original_words,
        "summary_word_count": summary_words,
        "original_sentence_count": original_sentences,
        "summary_sentence_count": summary_sentences,
        "compression_ratio": compression_ratio,
        "compression_percentage": round(compression_ratio * 100, 2),
        "reduction_percentage": reduction,
    }
