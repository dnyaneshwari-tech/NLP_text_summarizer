from __future__ import annotations

import math
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None

from config import DEFAULT_ABSTRACTIVE_MODEL


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_cache(func):
    if st is not None:
        return st.cache_resource(show_spinner=False)(func)
    return func


def calculate_generation_limits(
    text: str, summary_length: str | int, tokenizer, model
) -> tuple[int, int]:
    """Estimate minimum and maximum output tokens from source size and requested length."""
    ratios = {
        "short": (0.20, 0.30),
        "medium": (0.30, 0.45),
        "long": (0.40, 0.55),
    }
    normalized_length = str(summary_length).strip().lower()
    if normalized_length not in ratios:
        normalized_length = {"2": "short", "3": "medium", "5": "long"}.get(
            normalized_length, "medium"
        )

    encoded = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    shape = getattr(encoded, "shape", ())
    source_tokens = int(shape[-1]) if len(shape) > 1 else len(encoded)

    tokenizer_limit = getattr(tokenizer, "model_max_length", 1024)
    if not isinstance(tokenizer_limit, int) or tokenizer_limit > 1_000_000:
        tokenizer_limit = 1024
    source_tokens = min(source_tokens, tokenizer_limit)

    config = model.config
    decoder_limit = (
        getattr(config, "max_position_embeddings", None)
        or getattr(config, "max_decoder_position_embeddings", None)
        or getattr(config, "n_positions", None)
        or 1024
    )
    if not isinstance(decoder_limit, int) or decoder_limit > 1_000_000:
        decoder_limit = 1024
    output_limit = max(1, min(768, decoder_limit - 2))
    minimum_ratio, maximum_ratio = ratios[normalized_length]
    minimum_tokens = math.ceil(source_tokens * minimum_ratio)
    maximum_tokens = math.ceil(source_tokens * maximum_ratio)
    maximum_tokens = min(output_limit, max(1, maximum_tokens))
    minimum_tokens = min(maximum_tokens - 1, max(0, minimum_tokens))
    return minimum_tokens, maximum_tokens


@_maybe_cache
def load_summarization_model(model_name: str = DEFAULT_ABSTRACTIVE_MODEL):
    """Load and cache the tokenizer and model only when a summarization request is made."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = _get_device()
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as exc:  # pragma: no cover - surfaced in app UI
        raise RuntimeError(f"Unable to load model '{model_name}'. {exc}") from exc


def summarize_with_transformer(
    text: str,
    model_name: str = DEFAULT_ABSTRACTIVE_MODEL,
    summary_length: str | None = None,
    min_new_tokens: int | None = None,
    max_new_tokens: int | None = None,
    chunking_enabled: bool = False,
) -> str:
    """Generate an abstractive summary using a pretrained transformer model."""
    if not text or not text.strip():
        return "No text available for abstractive summarization."

    try:
        tokenizer, model, device = load_summarization_model(model_name)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    calculated_minimum, calculated_maximum = calculate_generation_limits(
        text,
        summary_length or "Medium",
        tokenizer,
        model,
    )
    if max_new_tokens is None:
        max_new_tokens = calculated_maximum
    else:
        max_new_tokens = max(1, min(int(max_new_tokens), 768))

    decoder_limit = getattr(model.config, "max_position_embeddings", None)
    if decoder_limit is None:
        decoder_limit = getattr(model.config, "n_positions", None)
    if isinstance(decoder_limit, int) and 1 < decoder_limit < 1_000_000:
        max_new_tokens = min(max_new_tokens, decoder_limit - 1)
    if min_new_tokens is None:
        min_new_tokens = calculated_minimum
    min_new_tokens = max(0, min(int(min_new_tokens), max_new_tokens - 1))

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=min(1024, tokenizer.model_max_length if tokenizer.model_max_length < 1_000_000 else 1024),
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
        )

    summary = tokenizer.decode(generated[0], skip_special_tokens=True)
    if not summary or not summary.strip():
        raise ValueError("The model produced an empty summary. Try a shorter input or extractive summarization.")
    return summary.strip()
