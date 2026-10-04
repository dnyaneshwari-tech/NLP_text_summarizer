from __future__ import annotations

import hashlib

import streamlit as st
import streamlit.components.v1 as components

from config import APP_DESCRIPTION, APP_TITLE, MAX_UPLOAD_BYTES
from src.document_loader import load_uploaded_document
from src.extractive_summarizer import extractive_summary
from src.long_document import summarize_long_document
from src.statistics import generate_summary_stats
from src.text_preprocessing import count_words, is_empty_text, split_into_sentences
from src.utils import get_default_sample_text, get_summary_length_value

try:
    from src.abstractive_summarizer import summarize_with_transformer
except Exception:  # pragma: no cover
    summarize_with_transformer = None


def initialize_session_state() -> None:
    defaults = {
        "input_mode": "Paste text",
        "input_text": "",
        "input_text_widget": "",
        "uploaded_text": "",
        "uploaded_file_name": "",
        "uploaded_file_identity": "",
        "method": "Extractive",
        "summary_length": "Medium",
        "target_sentences": 3,
        "chunking_enabled": True,
        "model_name": "facebook/bart-large-cnn",
        "generated_summary": "",
        "summary_stats": {},
        "status_message": "",
        "generate_summary": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_generated_summary() -> None:
    st.session_state["generated_summary"] = ""
    st.session_state["summary_stats"] = {}
    st.session_state["status_message"] = ""


def trigger_generate_summary() -> None:
    clear_generated_summary()
    st.session_state["generate_summary"] = True


def sync_input_text_widget() -> None:
    if "input_text_widget" in st.session_state:
        new_text = st.session_state["input_text_widget"]
        if new_text != st.session_state.get("input_text", ""):
            clear_generated_summary()
            st.session_state["input_text"] = new_text


def sync_input_mode_widget() -> None:
    if "input_mode_widget" in st.session_state:
        st.session_state["input_mode"] = st.session_state["input_mode_widget"]


def set_page_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-primary: #0B0F19;
            --bg-surface: rgba(255,255,255,0.03);
            --bg-surface-strong: rgba(255,255,255,0.05);
            --border-soft: rgba(255,255,255,0.08);
            --text-primary: #EAF2FF;
            --text-muted: rgba(234,242,255,0.72);
            --accent-blue: #3B82F6;
            --accent-violet: #8B5CF6;
            --shadow-soft: 0 20px 45px rgba(7, 10, 17, 0.42);
        }

        .stApp {
            background: var(--bg-primary);
            color: var(--text-primary);
        }

        header[data-testid="stHeader"] {
            background: #111827 !important;
            color: #EAF2FF;
        }

        header[data-testid="stHeader"] [data-testid="stToolbar"] {
            background: transparent !important;
        }

        .stApp::before {
            content: "";
            position: fixed;
            left: 50%;
            top: -120px;
            width: 720px;
            height: 720px;
            transform: translateX(-50%);
            background: radial-gradient(circle, rgba(59, 130, 246, 0.12), rgba(59, 130, 246, 0.04) 25%, transparent 60%);
            pointer-events: none;
            z-index: 0;
        }

        .stApp::after {
            content: "";
            position: fixed;
            right: -120px;
            bottom: -100px;
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(139, 92, 246, 0.1), rgba(139, 92, 246, 0.04) 25%, transparent 60%);
            pointer-events: none;
            z-index: 0;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            position: relative;
            z-index: 1;
            max-width: 1400px;
        }

        h1, h2, h3, h4, h5 {
            color: var(--text-primary);
            letter-spacing: -0.03em;
        }

        p, li, div, span, label {
            color: var(--text-primary);
        }

        .stMarkdown, .stCaption, .stDataFrame, .stFileUploaderLabel, .stTextInput, .stTextArea, .stNumberInput, .stSelectbox, .stCheckbox {
            color: var(--text-primary);
        }

        .stAlert, .stSuccess, .stInfo, .stWarning, .stError {
            border-radius: 14px;
            border: 1px solid var(--border-soft);
            background: rgba(255,255,255,0.02);
            color: var(--text-primary);
        }

        .st-key-document_extracted_notice [data-testid="stAlert"] {
            background: rgba(59,130,246,0.10) !important;
            border: 1px solid rgba(59,130,246,0.25) !important;
            color: #E0E7FF !important;
        }

        .st-key-document_extracted_notice [data-testid="stAlert"] * {
            color: #E0E7FF !important;
            -webkit-text-fill-color: #E0E7FF !important;
        }

        .feature-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-soft);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            min-height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: var(--shadow-soft);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
        }

        .feature-card h4 {
            margin-bottom: 0.45rem;
            color: var(--text-primary);
            font-size: 1.08rem;
            font-weight: 700;
        }

        .feature-card p {
            margin: 0;
            color: var(--text-muted);
            line-height: 1.55;
        }

        section[data-testid="stSidebar"] {
            background: rgba(9, 13, 21, 0.85);
            border-right: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            box-shadow: inset -1px 0 0 rgba(255,255,255,0.04);
        }

        [data-testid="stSidebar"] .stMarkdown {
            color: var(--text-primary);
        }

        [data-testid="stSidebar"] .stRadio > div {
            gap: 0.5rem;
        }

        [data-testid="stSidebar"] [role="radiogroup"] {
            gap: 0.35rem;
        }

        [data-testid="stSidebar"] [role="radio"] {
            border-radius: 12px;
            border: 1px solid transparent;
            padding: 0.6rem 0.7rem;
            background: transparent;
            color: var(--text-primary);
            transition: all 0.2s ease;
        }

        [data-testid="stSidebar"] [role="radio"]:hover {
            background: rgba(255,255,255,0.02);
            border-color: rgba(255,255,255,0.04);
        }

        [data-testid="stSidebar"] [role="radio"][aria-checked="true"] {
            background: linear-gradient(135deg, rgba(59,130,246,0.28), rgba(139,92,246,0.2));
            border: 1px solid rgba(118, 164, 255, 0.45);
            box-shadow: 0 0 0 1px rgba(59,130,246,0.14), 0 12px 30px rgba(59,130,246,0.12);
        }

        .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.08);
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-violet));
            color: white;
            font-weight: 600;
            box-shadow: 0 12px 20px rgba(59,130,246,0.18);
        }

        .stButton > button:hover {
            filter: brightness(1.04);
        }

        .stDownloadButton > button {
            border-radius: 12px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-violet));
            border: none;
            color: white;
            box-shadow: 0 12px 20px rgba(59,130,246,0.18);
        }

        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div,
        .stFileUploader > div {
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 12px;
            color: #F8FAFC;
        }

        .stTextArea > div > div > textarea,
        .stTextArea textarea,
        [data-testid="stTextArea"] textarea {
            background: #0F172A !important;
            color: #F8FAFC !important;
            caret-color: #F8FAFC;
            -webkit-text-fill-color: #F8FAFC !important;
            border: 1px solid rgba(148, 163, 184, 0.35) !important;
            border-radius: 12px !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }

        .stTextArea > div > div > textarea::placeholder,
        .stTextArea textarea::placeholder,
        [data-testid="stTextArea"] textarea::placeholder {
            color: #94A3B8 !important;
            opacity: 1;
        }

        .stTextArea > div > div > textarea:focus,
        .stTextArea textarea:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: rgba(96, 165, 250, 0.8) !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.18), inset 0 1px 0 rgba(255,255,255,0.04) !important;
        }

        .stApp:has([data-testid="stFileUploader"]) [data-testid="stFileUploaderDropzone"] {
            background: #182235 !important;
            border: 1px dashed rgba(59,130,246,0.35) !important;
            border-radius: 12px !important;
            color: #E2E8F0 !important;
        }

        .stApp:has([data-testid="stFileUploader"]) [data-testid="stFileUploaderDropzoneInstructions"],
        .stApp:has([data-testid="stFileUploader"]) [data-testid="stFileUploaderDropzoneInstructions"] span {
            color: #94A3B8 !important;
            -webkit-text-fill-color: #94A3B8 !important;
        }

        .stApp:has([data-testid="stFileUploader"]) [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"],
        .stApp:has([data-testid="stFileUploader"]) [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p {
            color: #E2E8F0 !important;
            -webkit-text-fill-color: #E2E8F0 !important;
        }

        .stApp:has([data-testid="stFileUploader"]) [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] {
            background: #22304A !important;
            border: 1px solid rgba(59,130,246,0.35) !important;
            border-radius: 8px !important;
            box-shadow: none !important;
            color: #F1F5F9 !important;
        }

        .stApp:has([data-testid="stFileUploader"]) [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] * {
            color: #F1F5F9 !important;
            -webkit-text-fill-color: #F1F5F9 !important;
        }

        .stApp:has([data-testid="stFileUploader"]) [data-testid="stFileUploader"] .stFileChip {
            background: #22304A;
            border: 1px solid rgba(148,163,184,0.18);
            color: #F1F5F9 !important;
        }

        .stApp:has([data-testid="stFileUploader"]) [data-testid="stFileUploader"] .stFileChip * {
            color: #F1F5F9 !important;
            -webkit-text-fill-color: #F1F5F9 !important;
        }

        .stApp:has([data-testid="stFileUploader"]) [data-testid="stExpander"] details {
            background: rgba(59,130,246,0.04);
            border: 1px solid rgba(148,163,184,0.15);
            border-radius: 12px;
        }

        .stApp:has([data-testid="stFileUploader"]) [data-testid="stExpander"] summary,
        .stApp:has([data-testid="stFileUploader"]) [data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] {
            background: #182235 !important;
            color: #F1F5F9 !important;
        }

        .stApp:has([data-testid="stFileUploader"]) [data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
            color: #F1F5F9 !important;
        }

        .stApp:has([data-testid="stFileUploader"]) [data-testid="stExpanderDetails"] {
            background: rgba(59,130,246,0.04);
        }

        .stApp:has([data-testid="stFileUploader"]) [data-testid="stExpanderDetails"] [data-testid="stText"] {
            box-sizing: border-box;
            display: block;
            width: 100%;
            background: #182235;
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 8px;
            color: #E2E8F0 !important;
            padding: 0.85rem 1rem;
        }

        .stApp:has([data-testid="stFileUploader"]) [data-testid="stExpanderDetails"] [data-testid="stText"] span {
            color: #E2E8F0 !important;
            -webkit-text-fill-color: #E2E8F0 !important;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }

        .stMetric {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            box-shadow: var(--shadow-soft);
        }

        .summarizer-shell {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            margin-top: 0.5rem;
        }

        .bento-shell {
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
        }

        .bento-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            box-shadow: 0 18px 38px rgba(7, 10, 17, 0.34);
            padding: 1.1rem 1.15rem;
            height: 100%;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
        }

        .bento-title {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: rgba(234,242,255,0.7);
            margin-bottom: 0.85rem;
        }

        .segment-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.6rem;
            margin-top: 0.3rem;
        }

        .segmented-control {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.5rem;
            padding: 0.38rem;
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
        }

        .segmented-control label {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 42px;
            border-radius: 12px;
            font-weight: 600;
            color: rgba(234,242,255,0.72);
            background: transparent;
            border: 1px solid transparent;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .segmented-control input {
            display: none;
        }

        .segmented-control input:checked + span {
            display: block;
            width: 100%;
            text-align: center;
            padding: 0.8rem 0.7rem;
            border-radius: 10px;
            background: linear-gradient(135deg, rgba(59,130,246,0.28), rgba(139,92,246,0.18));
            border: 1px solid rgba(118, 164, 255, 0.45);
            box-shadow: 0 0 0 1px rgba(59,130,246,0.14), 0 10px 30px rgba(59,130,246,0.12);
            color: var(--text-primary);
        }

        .st-key-summary_output {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            box-shadow: 0 18px 38px rgba(7, 10, 17, 0.34);
            padding: 1.1rem 1.15rem;
            min-height: 280px;
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
        }

        .summary-empty {
            border: 1px dashed rgba(255,255,255,0.1);
            border-radius: 14px;
            background: rgba(255,255,255,0.01);
            color: rgba(234,242,255,0.72);
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 180px;
            text-align: center;
            padding: 1rem;
        }

        .st-key-summary_output .summary-text {
            white-space: pre-wrap;
            line-height: 1.7;
            margin-bottom: 32px;
            color: var(--text-primary);
        }

        .summary-stats-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
        }

        .summary-stats-row .stMetric {
            min-height: 90px;
        }

        .settings-actions {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-top: 0.5rem;
        }

        .settings-actions .stButton > button {
            min-width: 180px;
        }

        .settings-footer-note {
            color: rgba(234,242,255,0.68);
            font-size: 0.86rem;
            margin-top: 0.5rem;
        }

        .nav-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f4c81;
        }

        .stApp:has(.settings-page-style-marker) {
            background: #111827;
            color: #F1F5F9;
        }

        .stApp:has(.settings-page-style-marker)::before {
            background: radial-gradient(circle, rgba(59,130,246,0.08), rgba(59,130,246,0.025) 25%, transparent 60%);
        }

        .stApp:has(.settings-page-style-marker)::after {
            background: radial-gradient(circle, rgba(139,92,246,0.06), rgba(139,92,246,0.02) 25%, transparent 60%);
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] {
            color: #F1F5F9;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] h1,
        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] h2,
        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] h3,
        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] h4,
        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] [data-testid="stMarkdownContainer"],
        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] [data-testid="stWidgetLabel"] {
            color: #F1F5F9;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] .bento-title {
            color: #CBD5E1;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            margin: 0 0 0.45rem;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] .bento-card {
            background: transparent;
            border: 0;
            border-radius: 0;
            box-shadow: none;
            padding: 0;
            height: auto;
            backdrop-filter: none;
            -webkit-backdrop-filter: none;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] label[data-testid="stRadioOption"] {
            border: 1px solid transparent;
            border-radius: 8px;
            color: #CBD5E1;
            padding: 0.4rem 0.6rem;
            transition: background 0.18s ease, border-color 0.18s ease;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] label[data-testid="stRadioOption"]:has(input:checked) {
            background: linear-gradient(135deg, rgba(59,130,246,0.14), rgba(139,92,246,0.09));
            border-color: rgba(148,163,184,0.18);
            color: #F1F5F9;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] input[type="text"],
        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] input[type="number"],
        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] [data-testid="stTextInput"] input,
        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] [data-testid="stNumberInput"] input {
            background: #182235 !important;
            border: 1px solid rgba(148,163,184,0.18) !important;
            border-radius: 8px !important;
            color: #F1F5F9 !important;
            -webkit-text-fill-color: #F1F5F9 !important;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] input::placeholder {
            color: #94A3B8 !important;
            -webkit-text-fill-color: #94A3B8 !important;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] input[type="text"]:focus,
        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] input[type="number"]:focus,
        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] [data-testid="stTextInput"] input:focus,
        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] [data-testid="stNumberInput"] input:focus {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 2px rgba(59,130,246,0.12) !important;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] [data-testid="stCheckbox"] input {
            accent-color: #3B82F6;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] [data-testid="stExpander"] details {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 8px;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] [data-testid="stExpander"] summary {
            background: #182235 !important;
            border: 1px solid rgba(148,163,184,0.18) !important;
            border-radius: 8px !important;
            color: #F1F5F9 !important;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] [data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
            color: #F1F5F9 !important;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #3B82F6, #8B5CF6);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 8px;
            box-shadow: 0 5px 14px rgba(59,130,246,0.14);
            color: #F1F5F9;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] .stButton > button[kind="secondary"] {
            background: #182235;
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 8px;
            box-shadow: none;
            color: #CBD5E1;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] .stDownloadButton > button {
            background: #182235;
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 8px;
            box-shadow: none;
            color: #F1F5F9;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] .stMetric {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 8px;
            box-shadow: none;
        }

        .stApp:has(.settings-page-style-marker) .st-key-summary_output {
            background: rgba(59,130,246,0.04);
            border: 1px solid rgba(148,163,184,0.15);
            border-radius: 8px;
            box-shadow: none;
            gap: 0.7rem;
            min-height: 0;
            padding: 1rem;
            backdrop-filter: none;
            -webkit-backdrop-filter: none;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] .summary-empty {
            background: transparent;
            border: 0;
            border-radius: 0;
            color: #94A3B8;
            display: block;
            min-height: 0;
            padding: 0;
            text-align: left;
        }

        .stApp:has(.settings-page-style-marker) .st-key-summary_output .summary-text {
            color: #F1F5F9;
        }

        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] .settings-footer-note,
        .stApp:has(.settings-page-style-marker) section[data-testid="stMain"] [data-testid="stCaptionContainer"] {
            color: #94A3B8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    st.title(APP_TITLE)
    st.caption(APP_DESCRIPTION)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class='feature-card'>
                <h4>Extractive NLP</h4>
                <p>Identify the most important sentences using transparent word-frequency scoring.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class='feature-card'>
                <h4>Abstractive NLP</h4>
                <p>Generate concise, fluent summaries with a transformer-based language model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class='feature-card'>
                <h4>Transformer Summarization</h4>
                <p>Generate fluent summaries with a pretrained Hugging Face model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class='feature-card'>
                <h4>Document Processing</h4>
                <p>Handle TXT, PDF, and DOCX uploads with extraction and text previews.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Workflow")
    st.markdown("1. Upload a file or paste text\n2. Choose a summarization method\n3. Select a summary length\n4. Review the result and download it")


def render_summary_output() -> None:
    summary_text = st.session_state.get("generated_summary", "")
    stats = st.session_state.get("summary_stats", {})

    with st.container(key="summary_output"):
        st.markdown("<div id='current-summary' class='bento-title'>Current Summary</div>", unsafe_allow_html=True)

        if not summary_text:
            st.markdown('<div class="summary-empty">Generate a summary to see the result here.</div>', unsafe_allow_html=True)
            return

        display_summary = " ".join(summary_text.split())
        st.markdown("<div class='summary-text'>" + display_summary + "</div>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Original words", stats.get("original_word_count", 0))
        with col2:
            st.metric("Summary words", stats.get("summary_word_count", 0))
        with col3:
            st.metric("Compression ratio", f"{stats.get('compression_percentage', 0):.1f}%")
        with col4:
            st.metric("Reduction", f"{stats.get('reduction_percentage', 0):.1f}%")

        st.caption(f"Processing status: {st.session_state.get('status_message', 'Ready')}")
        st.download_button(
            label="Download Summary",
            data=summary_text,
            file_name="summary.txt",
            mime="text/plain",
            key="download_summary_txt",
        )
        st.warning("Disclaimer: this summary is AI-generated and may omit context, important details, or contain minor errors. Please verify key facts against the original source.")


def render_summarizer() -> None:
    st.markdown('<div class="summarizer-shell">', unsafe_allow_html=True)

    render_header = st.container()
    with render_header:
        col_title, col_badge = st.columns([5, 1])
        with col_title:
            st.markdown("<h3 style='margin:0;'>Add content to summarize</h3>", unsafe_allow_html=True)
        with col_badge:
            st.markdown('<div class="summarizer-badge">AI Workspace</div>', unsafe_allow_html=True)

    input_panel = st.container()
    with input_panel:
        if "input_mode_widget" not in st.session_state:
            st.session_state["input_mode_widget"] = st.session_state.get("input_mode", "Paste text")

        mode = st.radio(
            "Choose an input mode",
            ["Paste text", "Upload document"],
            horizontal=True,
            key="input_mode_widget",
            on_change=sync_input_mode_widget,
        )

        if mode == "Paste text":
            if "input_text_widget" not in st.session_state:
                st.session_state["input_text_widget"] = st.session_state.get("input_text", "")

            col_load, col_clear = st.columns([1, 1])
            with col_load:
                if st.button("Load sample text", use_container_width=True):
                    sample = get_default_sample_text()
                    clear_generated_summary()
                    st.session_state["input_text"] = sample
                    st.session_state["input_text_widget"] = sample
            with col_clear:
                if st.button("Clear input", use_container_width=True):
                    clear_generated_summary()
                    st.session_state["input_text"] = ""
                    st.session_state["input_text_widget"] = ""

            st.markdown('<div class="plain-text-surface">', unsafe_allow_html=True)
            text_value = st.text_area(
                "",
                value=st.session_state.get("input_text", ""),
                key="input_text_widget",
                height=360,
                placeholder="Paste your text here...",
                on_change=sync_input_text_widget,
                label_visibility="collapsed",
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if text_value:
                st.caption(f"Word count: {count_words(text_value)}")
                st.caption(f"Sentence count: {len(split_into_sentences(text_value))}")
            else:
                st.info("Please paste some text or load the sample text to continue.")

        else:
            st.markdown('<div class="upload-dropzone">', unsafe_allow_html=True)
            st.markdown('<div class="upload-icon">📄</div>', unsafe_allow_html=True)
            st.markdown('<strong>Drag & drop your files here</strong>', unsafe_allow_html=True)
            st.markdown('<span>or browse from your computer</span>', unsafe_allow_html=True)
            st.markdown('<div class="formats">TXT • PDF • DOCX</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "",
                key="uploaded_document",
                type=["txt", "pdf", "docx"],
                help=f"Accepted file types: TXT, PDF, DOCX. Maximum size: {MAX_UPLOAD_BYTES / (1024 * 1024):.0f} MB.",
                label_visibility="collapsed",
            )

            if uploaded_file is not None:
                st.session_state["input_mode"] = "Upload document"
                file_bytes = uploaded_file.getvalue()
                file_identity = hashlib.sha256(
                    uploaded_file.name.encode("utf-8") + b"\0" + file_bytes
                ).hexdigest()
                if file_identity != st.session_state.get("uploaded_file_identity", ""):
                    clear_generated_summary()
                    st.session_state["uploaded_file_identity"] = file_identity
                    st.session_state["uploaded_text"] = ""
                    st.session_state["uploaded_file_name"] = ""
                    try:
                        extracted_text, original_name = load_uploaded_document(uploaded_file)
                        if not extracted_text.strip():
                            st.warning("This document contains no extractable text.")
                            return
                        st.session_state["uploaded_text"] = extracted_text
                        st.session_state["uploaded_file_name"] = original_name
                        with st.container(key="document_extracted_notice"):
                            st.success("Document extracted successfully.")
                    except Exception as exc:
                        st.error(f"Unable to extract text from this document: {exc}")
                        return
                elif not st.session_state.get("uploaded_text", "").strip():
                    st.warning("This document has no available extracted text. Please upload it again or choose another file.")
                    return

                st.caption(f"Selected file: {st.session_state.get('uploaded_file_name', uploaded_file.name)}")
                with st.expander("Preview extracted text"):
                    st.text(st.session_state["uploaded_text"][:4000])
                st.caption(f"Extracted word count: {count_words(st.session_state.get('uploaded_text', ''))}")
            else:
                if st.session_state.get("uploaded_text"):
                    st.caption(f"Selected file: {st.session_state.get('uploaded_file_name', 'previous upload')}")
                    with st.expander("Preview extracted text"):
                        st.text(st.session_state["uploaded_text"][:4000])

    st.markdown('</div>', unsafe_allow_html=True)


def render_settings() -> None:
    st.markdown('<div class="settings-page-style-marker"></div>', unsafe_allow_html=True)
    st.subheader("Summarization Settings & Result")

    st.markdown("<div class='bento-title'>Method</div>", unsafe_allow_html=True)
    method_options = ["Extractive NLP", "Abstractive / Transformer"]
    method_lookup = {"Extractive NLP": "Extractive", "Abstractive / Transformer": "Abstractive"}
    current_method_label = "Extractive NLP" if st.session_state.get("method", "Extractive") == "Extractive" else "Abstractive / Transformer"
    method_choice = st.radio(
        "",
        method_options,
        index=method_options.index(current_method_label),
        key="settings_method_radio",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state["method"] = method_lookup[method_choice]

    st.markdown("<div class='bento-title'>Summary Length</div>", unsafe_allow_html=True)
    length_options = ["Short", "Medium", "Long"]
    current_length = st.session_state.get("summary_length", "Medium")
    length_choice = st.radio(
        "",
        length_options,
        index=length_options.index(current_length) if current_length in length_options else 1,
        key="settings_summary_length_radio",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state["summary_length"] = length_choice

    with st.expander("Advanced configuration"):
        st.caption("The default transformer model is facebook/bart-large-cnn. A smaller model may be more suitable for limited-resource systems.")
        st.text_input(
            "Model name",
            value=st.session_state.get("model_name", "facebook/bart-large-cnn"),
            help="Change this only when you want to try a different Hugging Face summarization model.",
            key="model_name",
        )
        st.number_input(
            "Approximate target sentences (extractive only)",
            min_value=1,
            max_value=10,
            value=st.session_state.get("target_sentences", 3),
            step=1,
            key="target_sentences",
        )
        st.checkbox("Enable chunking for long documents", value=st.session_state.get("chunking_enabled", True), key="chunking_enabled")

    st.markdown("<div class='bento-title'>Generate</div>", unsafe_allow_html=True)
    st.button("Generate Summary", type="primary", use_container_width=True, on_click=trigger_generate_summary)
    st.markdown("<div class='settings-footer-note'>Uses the current method and summary length.</div>", unsafe_allow_html=True)

    render_summary_output()


def get_active_text() -> str:
    if st.session_state.get("input_mode", "Paste text") == "Upload document":
        return st.session_state.get("uploaded_text", "")
    return st.session_state.get("input_text", "")


def compute_summary() -> tuple[str, dict, str]:
    text_source = get_active_text()

    if not text_source or is_empty_text(text_source):
        raise ValueError("Please enter text or upload a valid document before generating a summary.")

    method = st.session_state.get("method", "Extractive")
    summary_length = st.session_state.get("summary_length", "Medium")
    target_sentences = st.session_state.get("target_sentences", 3)
    chunking_enabled = st.session_state.get("chunking_enabled", True)
    model_name = st.session_state.get("model_name", "facebook/bart-large-cnn")

    if method == "Extractive":
        chosen_length = get_summary_length_value(summary_length)
        summary = extractive_summary(text_source, summary_length=chosen_length, target_sentences=int(target_sentences))
        stats = generate_summary_stats(text_source, summary)
        status = "Extractive summarization completed."
        return summary, stats, status

    if summarize_with_transformer is None:
        raise RuntimeError("The transformer summarizer is unavailable in this environment.")

    summary_target = get_summary_length_value(summary_length)
    try:
        if len(split_into_sentences(text_source)) > 8 and chunking_enabled:
            from src.abstractive_summarizer import load_summarization_model

            tokenizer, _, _ = load_summarization_model(model_name)
            tokenizer_limit = getattr(tokenizer, "model_max_length", 1024)
            if not isinstance(tokenizer_limit, int) or tokenizer_limit > 1_000_000:
                tokenizer_limit = 1024
            summary = summarize_long_document(
                text_source,
                lambda chunk, length: summarize_with_transformer(
                    chunk,
                    model_name=model_name,
                    summary_length=str(length),
                ),
                tokenizer,
                max_tokens=min(1024, tokenizer_limit),
                target_summary_length=summary_target,
            )
        else:
            summary = summarize_with_transformer(
                text_source,
                model_name=model_name,
                summary_length=str(summary_target),
            )
        stats = generate_summary_stats(text_source, summary)
        status = "Abstractive summarization completed."
        return summary, stats, status
    except Exception as exc:
        raise RuntimeError(f"The transformer model could not produce a summary: {exc}") from exc


def main() -> None:
    initialize_session_state()
    set_page_style()
    nav = st.sidebar.radio("Navigation", ["Home", "Summarizer", "Summarization Settings & Result"])
    summary_generated = False

    if st.session_state.get("generate_summary"):
        clear_generated_summary()
        try:
            with st.spinner("Generating summary... Please wait while your document is being summarized."):
                summary_text, stats, status_text = compute_summary()
            st.session_state["generated_summary"] = summary_text
            st.session_state["summary_stats"] = stats
            st.session_state["status_message"] = status_text
            st.session_state["generate_summary"] = False
            summary_generated = True
        except Exception as exc:
            st.session_state["generate_summary"] = False
            st.error(str(exc))

    if nav == "Home":
        render_home()
    elif nav == "Summarizer":
        render_summarizer()
    else:
        render_settings()

    if summary_generated and nav == "Summarization Settings & Result":
        components.html(
            """
            <script>
            setTimeout(() => {
                const target = window.parent.document.getElementById('current-summary');
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 120);
            </script>
            """,
            height=0,
        )


if __name__ == "__main__":
    main()
