from src.long_document import split_text_into_chunks, summarize_long_document


class WordTokenizer:
    model_max_length = 3

    def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
        return {"input_ids": text.split()}

    def num_special_tokens_to_add(self, pair=False):
        return 0


def test_long_sentences_are_split_into_token_safe_word_chunks():
    text = "One two three four five six. Seven eight nine."

    chunks = split_text_into_chunks(text, WordTokenizer(), max_tokens=3)

    assert chunks
    assert all(len(chunk.split()) <= 3 for chunk in chunks)
    assert " ".join(chunks).replace(". ", " ").split() == text.replace(". ", " ").split()


def test_intermediate_summaries_are_refined_before_final_pass():
    calls = 0

    def summarizer(text, target_length):
        nonlocal calls
        calls += 1
        if calls <= 2:
            return text
        if calls <= 4:
            return "brief"
        return "final summary"

    result = summarize_long_document(
        "One two three. Four five six.",
        summarizer,
        WordTokenizer(),
        max_tokens=3,
    )

    assert result == "final summary"
    assert calls == 5
