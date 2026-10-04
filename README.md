# AI-Powered Text & Document Summarizer

This project is a Streamlit-based NLP application that summarizes pasted text and uploaded documents using both extractive and transformer-based abstractive methods. It is designed as an undergraduate AI and data science portfolio project and emphasizes practical NLP workflows, document extraction, and long-document handling.

## Project Overview

The application supports:

- Direct text input
- TXT, PDF, and DOCX uploads
- Extractive summarization using sentence scoring and word-frequency analysis
- Abstractive summarization using a pretrained Hugging Face transformer model
- Long-document summarization through chunking and second-pass summarization
- Summary statistics and download support

The app is meant for educational and demonstration purposes. Users should verify any important factual claims against the original document, especially for research or academic use.

## Features

- Extractive NLP summarization
- Transformer-based summarization with a configurable model
- PDF, DOCX, and TXT document processing
- Long-document chunking for improved transformer handling
- Summary length controls
- Word and sentence statistics
- Downloadable summary export

## Technology Stack

- Python 3.11+
- Streamlit
- Hugging Face Transformers
- PyTorch
- NLTK
- PyMuPDF (fitz)
- python-docx
- Pandas
- Plotly
- pytest

## Project Structure

```text
nlp_document_summarizer/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── config.py
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── text_preprocessing.py
│   ├── extractive_summarizer.py
│   ├── abstractive_summarizer.py
│   ├── long_document.py
│   ├── statistics.py
│   └── utils.py
├── tests/
│   ├── test_document_loader.py
│   ├── test_preprocessing.py
│   └── test_extractive_summarizer.py
└── .streamlit/
    └── config.toml
```

## Extractive Summarization

Extractive summarization selects the most important sentences from the original text instead of generating new wording. This project uses a transparent frequency-based method:

1. Split the text into sentences.
2. Normalize words and remove common stop words.
3. Count the frequency of meaningful words.
4. Score each sentence according to the importance of its words.
5. Select the best sentences and keep their original order.

This method is easy to understand and useful for short to medium-length documents.

## Transformer-Based Abstractive Summarization

Abstractive summarization uses a pretrained text-to-text model such as BART or T5 to generate a new summary. Rather than selecting sentences verbatim, the model creates condensed text in a more natural style.

This project uses Hugging Face Transformers and defaults to `facebook/bart-large-cnn`, which is a practical general-purpose model for English text summarization. The model is only loaded when needed and may require a compatible PyTorch build and enough RAM or VRAM.

## Long-Document Chunking

Large documents can exceed the token limit of transformer models. The project handles this with a chunking workflow:

1. Tokenize the document.
2. Split it into manageable chunks while preserving sentence boundaries where possible.
3. Summarize each chunk.
4. Combine the chunk summaries.
5. Run a second pass if the combined result is still too long.

Chunking can sometimes lose relationships between distant parts of a document, so users should treat the result as a helpful overview rather than a perfect abstraction of every detail.

## Installation Instructions (Windows PowerShell)

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

If the default PyTorch wheel is unsuitable for your computer, install a compatible build from the official PyTorch website, then reinstall the project requirements. The project does not hardcode a machine-specific install command into the requirements file.

## Running the App

Start the Streamlit server:

```powershell
streamlit run app.py
```

## Running Tests

```powershell
pytest
```

## Model Download and Initialization

When the app first loads a transformer model, it will download the model from Hugging Face if it is not already cached locally. If the model download is slow or fails, the app will show a clear error and allow the user to use extractive summarization instead.

## Streamlit Community Cloud Deployment

1. Create a GitHub repository for the project.
2. Upload the project files without including `.venv` or large model cache directories.
3. Connect the repository to Streamlit Community Cloud.
4. Select `app.py` as the main file.
5. Deploy the project and review the logs.

### Deployment notes

- Large transformer models may be slow or exceed memory limits.
- Community Cloud may not provide a GPU.
- Model downloads may take time on initial deployment.
- The app should be configured to load a model from a fresh environment.
- If required, a smaller model can be set in the configuration for lower-resource hosting.
- Do not store API keys or secrets in GitHub.

## Known Limitations

- Extractive summarization is simple and transparent, not a deep semantic model.
- Abstractive summaries can miss important context or make small factual errors.
- Scanned PDF text requires OCR, which is not included in this version.
- Chunking can lose context between distant sections in very long documents.
- Transformer performance depends on local hardware and internet access.

## Future Improvements

- Add OCR for scanned PDFs using Tesseract or equivalent.
- Add keyword extraction and topic summaries.
- Add multilingual model support.
- Add support for larger long-document summarization strategies.
- Add a summary comparison panel for extractive versus abstractive outputs.

## Repo and Deployment Readiness

This project is ready for local development and deployment through Streamlit Community Cloud. It follows a standard Python package layout and includes basic tests and documentation.
