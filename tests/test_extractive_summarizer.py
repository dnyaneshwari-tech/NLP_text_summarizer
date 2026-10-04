from src.extractive_summarizer import extractive_summary


def test_empty_input():
    result = extractive_summary("   \n\n", 2)
    assert "Not enough text" in result


def test_one_sentence_input():
    result = extractive_summary("Machine learning helps summarize documents.", 2)
    assert "Machine learning helps summarize documents." in result


def test_multi_sentence_input():
    text = (
        "Artificial intelligence helps automate many tasks. "
        "Data science turns raw information into insight. "
        "NLP systems can analyze text and identify important ideas. "
        "These models are helpful for summarization tasks."
    )
    summary = extractive_summary(text, 2)
    assert len(summary.split()) > 0
    assert "Artificial intelligence" in summary


def test_requested_summary_length():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    result = extractive_summary(text, 2)
    assert len(result.split(". ")) <= 2


def test_sentence_order_preserved():
    text = "First topic is important. Second topic is also relevant. Third topic closes the paragraph."
    summary = extractive_summary(text, 2)
    first_index = summary.find("First topic")
    second_index = summary.find("Second topic")
    assert first_index < second_index


def test_duplicate_prevention():
    text = "Repeat the idea. Repeat the idea. Another point matters."
    summary = extractive_summary(text, 2)
    assert summary.count("Repeat the idea") == 1
