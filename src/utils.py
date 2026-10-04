from __future__ import annotations


def get_summary_length_value(length_name: str) -> int:
    mapping = {"Short": 2, "Medium": 3, "Long": 5}
    return mapping.get(length_name, 3)


def get_default_sample_text() -> str:
    return (
        "Artificial intelligence is transforming the way organizations process large amounts of information. "
        "Modern systems can extract relevant patterns from text, identify key ideas, and organize content for "
        "faster review. This is useful for research, education, and decision support because large documents are "
        "difficult to read in full. A good summary should preserve the main conclusions while making the content "
        "shorter and easier to understand. Summarization systems often combine rule-based extraction with neural "
        "language models to improve readability and relevance."
    )


def ensure_positive_int(value, default: int = 3) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)
