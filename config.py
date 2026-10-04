from __future__ import annotations

DEFAULT_SUMMARY_MODEL = "facebook/bart-large-cnn"
ALT_SUMMARY_MODEL = "t5-small"
MAX_UPLOAD_SIZE_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

APP_TITLE = "AI-Powered Text & Document Summarizer"
APP_DESCRIPTION = (
    "A lightweight NLP web app for extracting text from documents and generating useful "
    "summaries using extractive and transformer-based methods."
)

# Use a conservative default for local demonstration. The model is loaded only when needed.
DEFAULT_ABSTRACTIVE_MODEL = DEFAULT_SUMMARY_MODEL

# This can be overridden by an environment variable or a future config file.
MODEL_NAME = DEFAULT_ABSTRACTIVE_MODEL
