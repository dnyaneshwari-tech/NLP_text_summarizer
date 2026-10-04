from __future__ import annotations

import re
from collections import Counter

from nltk.corpus import stopwords

from src.text_preprocessing import normalize_text, split_into_sentences


def _ensure_stopwords() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        import nltk

        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def extractive_summary(text: str, summary_length: int = 3, target_sentences: int | None = None) -> str:
    """Build a frequency-based extractive summary while keeping sentence order."""
    cleaned = normalize_text(text)
    if not cleaned or len(cleaned.strip()) < 3:
        return "Not enough text to summarize. Please provide a longer document or paragraph."

    sentences = split_into_sentences(cleaned)
    if not sentences:
        return "Not enough text to summarize. Please provide a longer document or paragraph."
    if len(sentences) == 1:
        return sentences[0]

    requested = target_sentences if target_sentences is not None else summary_length
    if requested is None or requested < 1:
        requested = 1
    requested = min(requested, len(sentences))

    stop_words = _ensure_stopwords()
    word_frequencies: Counter[str] = Counter()
    for sentence in sentences:
        tokens = [token.lower() for token in re.findall(r"\b[a-zA-Z]+\b", sentence)]
        for token in tokens:
            if token in stop_words or len(token) <= 2:
                continue
            word_frequencies[token] += 1

    if not word_frequencies:
        word_frequencies = Counter(
            token.lower() for sentence in sentences for token in re.findall(r"\b[a-zA-Z]+\b", sentence)
        )

    sentence_scores: list[tuple[int, float, str]] = []
    for idx, sentence in enumerate(sentences):
        tokens = [token.lower() for token in re.findall(r"\b[a-zA-Z]+\b", sentence)]
        if not tokens:
            continue
        score = sum(word_frequencies.get(token, 0) for token in tokens if token not in stop_words)
        sentence_scores.append((idx, score, sentence))

    if not sentence_scores:
        return sentences[0]

    top_sentences = sorted(sentence_scores, key=lambda item: (item[1], -item[0]), reverse=True)
    chosen: list[str] = []
    seen: set[str] = set()
    for _, _, sentence in top_sentences:
        normalized_sentence = " ".join(sentence.split())
        if normalized_sentence in seen:
            continue
        chosen.append(sentence)
        seen.add(normalized_sentence)
        if len(chosen) >= requested:
            break

    if not chosen:
        return sentences[0]

    chosen_normalized = {" ".join(sentence.split()) for sentence in chosen}
    final_sentences: list[str] = []
    emitted: set[str] = set()
    for sentence in sentences:
        normalized_sentence = " ".join(sentence.split())
        if normalized_sentence in chosen_normalized and normalized_sentence not in emitted:
            final_sentences.append(sentence)
            emitted.add(normalized_sentence)

    if len(final_sentences) < len(chosen):
        final_sentences = chosen[:]

    final_summary = " ".join(final_sentences)
    if not final_summary.strip():
        return sentences[0]

    return final_summary


def extractive_summary_with_length(text: str, summary_length: int = 3) -> str:
    """Small compatibility wrapper for explicit summary length settings."""
    return extractive_summary(text, summary_length=summary_length)
