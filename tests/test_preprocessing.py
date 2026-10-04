from src.text_preprocessing import (
    clean_whitespace,
    count_words,
    is_empty_text,
    normalize_text,
    split_into_sentences,
)


def test_whitespace_cleanup():
    text = "Hello     world\n\nThis   has   extra  spacing."
    cleaned = clean_whitespace(text)
    assert "Hello world" in cleaned
    assert "  " not in cleaned


def test_empty_text():
    assert is_empty_text("   \n\n") is True
    assert is_empty_text("A short sentence.") is False


def test_word_counting():
    text = "This is a short sentence. Another one follows."
    assert count_words(text) == 8


def test_sentence_splitting():
    text = "First sentence. Second sentence! Third? Fourth sentence."
    sentences = split_into_sentences(text)
    assert len(sentences) == 4
    assert sentences[0].startswith("First")
    assert sentences[-1].startswith("Fourth")


def test_normalize_text():
    text = "  hello   WORLD\n\n  ".strip()
    normalized = normalize_text(text)
    assert "hello" in normalized.lower()
    assert "WORLD" in normalized
