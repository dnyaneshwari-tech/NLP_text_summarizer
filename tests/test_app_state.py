import streamlit as st

from app import (
    compute_summary,
    get_active_text,
    sync_input_mode_widget,
    sync_input_text_widget,
)


def test_compute_summary_uses_session_input_text():
    st.session_state.clear()
    st.session_state["input_mode"] = "Paste text"
    st.session_state["input_text"] = (
        "Artificial intelligence is transforming healthcare by helping doctors detect disease earlier. "
        "Machine learning models can review patient records, identify patterns, and support clinical decisions. "
        "These systems are becoming more useful because they can process large amounts of information quickly."
    )
    st.session_state["method"] = "Extractive"
    st.session_state["summary_length"] = "Medium"
    st.session_state["target_sentences"] = 2
    st.session_state["chunking_enabled"] = True

    summary, stats, status = compute_summary()

    assert isinstance(summary, str)
    assert len(summary.strip()) > 0
    assert "Artificial intelligence" in summary or "Machine learning" in summary
    assert "original_word_count" in stats
    assert "summarization" in status.lower()


def test_compute_summary_uses_uploaded_document_text():
    st.session_state.clear()
    st.session_state["input_mode"] = "Upload document"
    st.session_state["uploaded_text"] = (
        "Medical imaging is becoming more accurate with deep learning systems. "
        "These models process large image datasets to detect small anomalies that may be difficult for clinicians to notice. "
        "The results support faster diagnosis and better patient monitoring."
    )
    st.session_state["uploaded_file_name"] = "notes.pdf"
    st.session_state["method"] = "Extractive"
    st.session_state["summary_length"] = "Medium"
    st.session_state["target_sentences"] = 2
    st.session_state["chunking_enabled"] = True

    summary, stats, status = compute_summary()

    assert isinstance(summary, str)
    assert len(summary.strip()) > 0
    assert "Medical imaging" in summary or "deep learning" in summary.lower()
    assert "original_word_count" in stats
    assert "summarization" in status.lower()


def test_input_mode_widget_syncs_persistent_document_mode():
    st.session_state.clear()
    st.session_state["input_mode"] = "Paste text"
    st.session_state["input_mode_widget"] = "Upload document"
    st.session_state["uploaded_text"] = "Extracted text from the selected document."

    sync_input_mode_widget()

    assert st.session_state["input_mode"] == "Upload document"
    assert get_active_text() == "Extracted text from the selected document."


def test_new_pasted_text_clears_previous_summary():
    st.session_state.clear()
    st.session_state["input_text"] = "Old source text"
    st.session_state["input_text_widget"] = "New source text"
    st.session_state["generated_summary"] = "Old summary"
    st.session_state["summary_stats"] = {"summary_word_count": 2}
    st.session_state["status_message"] = "Previous run completed"

    sync_input_text_widget()

    assert st.session_state["input_text"] == "New source text"
    assert st.session_state["generated_summary"] == ""
    assert st.session_state["summary_stats"] == {}
    assert st.session_state["status_message"] == ""
