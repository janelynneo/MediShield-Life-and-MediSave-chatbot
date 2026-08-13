"""
Statement Upload page — for end users to upload and analyze their own
MediShield Life / MediSave statements.

Flow: Upload file → Cloudflare OCR → PII masking → LLM explanation
"""

import streamlit as st
from dotenv import load_dotenv

# Ensure .env is loaded for local dev
from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env")

from utils.auth import verify_user, get_role

# ── Auth ──────────────────────────────────────────────────────────────────────
def require_auth():
    if not st.session_state.get("is_logged_in", False):
        st.error("Please log in to access this page.")
        st.stop()

require_auth()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Statement Upload — MediShield Assistant",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Statement Analyzer")
st.caption("Upload your MediShield Life or MediSave statement — we'll explain what it means.")

# ── File upload ────────────────────────────────────────────────────────────────
SUPPORTED_FORMATS = {
    "pdf": "PDF document",
    "docx": "Word document",
    "png": "PNG image",
    "jpg": "JPEG image",
    "jpeg": "JPEG image",
    "webp": "WebP image",
    "tiff": "TIFF image",
    "bmp": "BMP image",
    "gif": "GIF image",
    "txt": "Text file",
    "md": "Markdown file",
}

accepted_exts = list(SUPPORTED_FORMATS.keys())
uploaded_file = st.file_uploader(
    "Upload your statement",
    type=accepted_exts,
    help="Supported: PDF, DOCX, PNG, JPG, WEBP, TIFF, BMP, GIF, TXT, MD",
)

if not uploaded_file:
    st.info("👆 Upload a file above to get started.")
    st.markdown(
        "**How this works:**\n"
        "1. Upload your MediShield Life claim statement or MediSave transaction record\n"
        "2. We extract the text using OCR (for images/scanned PDFs)\n"
        "3. Personal information is automatically masked (NRIC, name, phone, etc.)\n"
        "4. You'll receive a plain-English explanation of your statement\n\n"
        "⚠️ Your file is processed securely and is never stored on our servers."
    )
    st.stop()

# ── Process file ──────────────────────────────────────────────────────────────
if "statement_result" not in st.session_state:
    st.session_state.statement_result = None
if "statement_masked_text" not in st.session_state:
    st.session_state.statement_masked_text = None
if "statement_processed_filename" not in st.session_state:
    st.session_state.statement_processed_filename = None

# ── Clear old result if new file uploaded ─────────────────────────────────────
if (
    uploaded_file
    and st.session_state.statement_processed_filename != uploaded_file.name
):
    st.session_state.statement_result = None
    st.session_state.statement_masked_text = None
    st.session_state.statement_processed_filename = uploaded_file.name

# ── Process file ────────────────────────────────────────────────────────────────
if uploaded_file and st.session_state.statement_result is None:
    with st.spinner("Extracting text from your document…"):

        from utils.statement_analyzer import extract_text, analyze_statement
        from utils.pii_masker import mask_all

        file_bytes = uploaded_file.read()
        filename = uploaded_file.name

        # 1. Extract text
        raw_text = extract_text(file_bytes, filename)

        if not raw_text or len(raw_text.strip()) < 20:
            st.error(
                "❌ Could not extract readable text from this file. "
                "Please ensure the document is a valid DOCX file and try again."
            )
            st.stop()

        # 2. Validate content — block potential prompt injection in uploaded documents
        import re
        injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+(instructions?|system)",
            r"forget\s+(all\s+)?instructions",
            r"disregard\s+(all\s+)?(your\s+)?instructions",
            r"you\s+are\s+now\s+",
            r"as\s+an\s+AI",
            r"pretend\s+you\s+are",
            r"system\s*:\s*",
            r"instruction\s*:\s*",
            r"delimiter\s*:",
            r"---+\s*$",
            r"^>>>\s*",
            r"^<\|<\|>",
        ]
        combined = "|".join(injection_patterns)
        if re.search(combined, raw_text, re.IGNORECASE):
            st.error(
                "❌ The uploaded document contains content that may indicate an injection attempt "
                "and cannot be processed. Please ensure your document is a genuine statement."
            )
            st.stop()

        # 3. Mask PII
        masked_text = mask_all(raw_text)
        st.session_state.statement_masked_text = masked_text

        # 4. Analyze with LLM
        with st.spinner("Analyzing your statement…"):
            explanation = analyze_statement(raw_text, masked_text)

        st.session_state.statement_result = explanation

# ── Show result ────────────────────────────────────────────────────────────────
if st.session_state.statement_result:
    st.markdown("---")
    st.markdown("### 📋 Explanation")

    # Show masked text preview (collapsible)
    if st.session_state.statement_masked_text:
        with st.expander("🔍 See masked text (what we sent to the AI)"):
            st.text_area(
                "Masked statement text",
                st.session_state.statement_masked_text,
                height=200,
                disabled=True,
                label_visibility="collapsed",
            )

    # Show explanation
    st.markdown(st.session_state.statement_result)

# ── Copy button ────────────────────────────────────────────────────────────────
st.markdown("---")
col_copy, _ = st.columns([1, 3])
with col_copy:
    if st.button("📋 Copy Explanation", use_container_width=True):
        st.code(st.session_state.statement_result, language=None)
        st.success("✅ Explanation copied! Select all text (Ctrl+A / Cmd+A) and copy.")
