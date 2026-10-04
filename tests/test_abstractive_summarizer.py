import pytest
import torch
from types import SimpleNamespace

from src import abstractive_summarizer


class FakeTokenizer:
    model_max_length = 512

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, tokens, skip_special_tokens=True):
        return "Generated summary"


class FakeModel:
    def __init__(self):
        self.generation_kwargs = None
        self.config = SimpleNamespace(max_position_embeddings=512)

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return torch.tensor([[4, 5]])


@pytest.mark.parametrize("summary_length", ["Short", "Medium", "Long", "3"])
def test_transformer_summary_accepts_ui_summary_lengths(monkeypatch, summary_length):
    tokenizer = FakeTokenizer()
    model = FakeModel()
    monkeypatch.setattr(
        abstractive_summarizer,
        "load_summarization_model",
        lambda model_name: (tokenizer, model, torch.device("cpu")),
    )

    summary = abstractive_summarizer.summarize_with_transformer(
        "A source sentence for the mocked summarizer.",
        summary_length=summary_length,
    )

    assert summary == "Generated summary"
    assert isinstance(model.generation_kwargs["max_new_tokens"], int)
    assert model.generation_kwargs["max_new_tokens"] >= model.generation_kwargs["min_new_tokens"]


def test_generation_length_scales_with_source_and_selected_length():
    class CountingTokenizer:
        model_max_length = 2048

        def __call__(self, text, **kwargs):
            token_count = int(len(text.split()) * 1.2)
            return {"input_ids": list(range(token_count))}

    tokenizer = CountingTokenizer()
    model = FakeModel()
    model.config.max_position_embeddings = 2048
    source_text = "source " * 350

    short_min, short_max = abstractive_summarizer.calculate_generation_limits(
        source_text, "Short", tokenizer, model
    )
    medium_min, medium_max = abstractive_summarizer.calculate_generation_limits(
        source_text, "Medium", tokenizer, model
    )
    long_min, long_max = abstractive_summarizer.calculate_generation_limits(
        source_text, "Long", tokenizer, model
    )

    assert short_min < medium_min < long_min
    assert short_max < medium_max < long_max
    assert (short_min, short_max) == (84, 126)
    assert (medium_min, medium_max) == (126, 189)
    assert (long_min, long_max) == (168, 232)

    longer_min, longer_max = abstractive_summarizer.calculate_generation_limits(
        "source " * 1000, "Long", tokenizer, model
    )
    assert longer_min > long_min
    assert longer_max > long_max


def test_generation_length_respects_decoder_limit():
    class CountingTokenizer:
        model_max_length = 4096

        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(3000))}

    minimum_tokens, maximum_tokens = abstractive_summarizer.calculate_generation_limits(
        "source", "Long", CountingTokenizer(), FakeModel()
    )

    assert maximum_tokens == 510
    assert minimum_tokens < maximum_tokens


def test_long_generation_constraints_reach_model_generate(monkeypatch):
    class CountingTokenizer(FakeTokenizer):
        model_max_length = 1024

        def __call__(self, text, **kwargs):
            token_count = int(len(text.split()) * 1.2)
            return {"input_ids": torch.arange(token_count).unsqueeze(0)}

    tokenizer = CountingTokenizer()
    model = FakeModel()
    model.config.max_position_embeddings = 1024
    monkeypatch.setattr(
        abstractive_summarizer,
        "load_summarization_model",
        lambda model_name: (tokenizer, model, torch.device("cpu")),
    )

    abstractive_summarizer.summarize_with_transformer(
        "source " * 350,
        summary_length="Long",
    )

    assert model.generation_kwargs["min_new_tokens"] == 168
    assert model.generation_kwargs["max_new_tokens"] == 232
