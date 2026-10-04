from __future__ import annotations

from typing import Callable

from transformers import AutoTokenizer

from src.text_preprocessing import split_into_sentences


def _estimate_token_count(text: str, tokenizer) -> int:
    tokens = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    return len(tokens)


def split_text_into_chunks(text: str, tokenizer, max_tokens: int = 1024) -> list[str]:
    """Create transformer-safe chunks while preferring sentence boundaries."""
    if not text or not text.strip():
        return []

    try:
        special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
    except (AttributeError, TypeError):
        special_token_count = 0
    token_budget = max(1, max_tokens - special_token_count)

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        sentence_parts = [sentence]
        if _estimate_token_count(sentence, tokenizer) > token_budget:
            sentence_parts = []
            current_part: list[str] = []
            for word in sentence.split():
                candidate = " ".join([*current_part, word])
                if _estimate_token_count(candidate, tokenizer) <= token_budget:
                    current_part.append(word)
                    continue
                if current_part:
                    sentence_parts.append(" ".join(current_part))
                    current_part = [word]
                elif _estimate_token_count(word, tokenizer) > token_budget:
                    raise ValueError(
                        "A single word exceeds the model input token limit and cannot be safely chunked."
                    )
                else:
                    current_part = [word]
            if current_part:
                sentence_parts.append(" ".join(current_part))

        for part in sentence_parts:
            candidate = f"{current_chunk} {part}".strip()
            if current_chunk and _estimate_token_count(candidate, tokenizer) > token_budget:
                chunks.append(current_chunk)
                current_chunk = part
            else:
                current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def summarize_long_document(
    text: str,
    summarizer_fn: Callable[[str, int], str],
    tokenizer,
    max_tokens: int = 1024,
    target_summary_length: int = 3,
) -> str:
    """Summarize a document in chunks and combine intermediate results."""
    if not text or not text.strip():
        return "Not enough text to summarize."

    chunks = split_text_into_chunks(text, tokenizer, max_tokens=max_tokens)
    if not chunks:
        return "Not enough text to summarize."
    if len(chunks) == 1:
        return summarizer_fn(chunks[0], target_summary_length)

    intermediate_summaries: list[str] = []
    for chunk in chunks:
        summary = summarizer_fn(chunk, max(1, min(3, target_summary_length)))
        if summary and summary.strip():
            intermediate_summaries.append(summary)

    combined = " \n".join(intermediate_summaries).strip()
    if not combined:
        return "Not enough text to summarize."

    try:
        special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
    except (AttributeError, TypeError):
        special_token_count = 0
    token_budget = max(1, max_tokens - special_token_count)

    for _ in range(3):
        combined_token_count = _estimate_token_count(combined, tokenizer)
        if combined_token_count <= token_budget:
            return summarizer_fn(combined, target_summary_length)

        previous_token_count = combined_token_count
        refinement_chunks = split_text_into_chunks(combined, tokenizer, max_tokens=max_tokens)
        refined_summaries = [
            summarizer_fn(chunk, max(1, min(3, target_summary_length)))
            for chunk in refinement_chunks
        ]
        combined = " ".join(summary.strip() for summary in refined_summaries if summary.strip())
        if not combined:
            return "Not enough text to summarize."
        if _estimate_token_count(combined, tokenizer) >= previous_token_count:
            raise RuntimeError("The model could not reduce the intermediate summaries to its input limit.")

    if _estimate_token_count(combined, tokenizer) <= token_budget:
        return summarizer_fn(combined, target_summary_length)
    raise RuntimeError("Intermediate summaries remain too long after three refinement passes.")
