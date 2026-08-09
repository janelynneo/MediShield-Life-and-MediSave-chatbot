"""
Statement analyzer — OCR + PII masking + LLM explanation.

Uses Cloudflare Workers AI for OCR on uploaded files (images + PDFs).
"""

import io
import os
import base64
import json as _json

import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from utils.pii_masker import mask_all

BASE_DIR = __file__.parent.parent
load_dotenv(BASE_DIR / ".env")


# ---------------------------------------------------------------------------
# Secrets helpers
# ---------------------------------------------------------------------------

def _get_secret(name: str) -> str:
    """Get a secret from Streamlit secrets (cloud) or env var (local)."""
    try:
        import streamlit as st
        return st.secrets.get(name, os.environ.get(name, ""))
    except ImportError:
        return os.environ.get(name, "")


# ---------------------------------------------------------------------------
# File text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF using PyPDF2."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception:
        return ""


def extract_text_from_image(file_bytes: bytes) -> str:
    """
    Extract text from an image using Cloudflare Workers AI OCR.

    Works with: PNG, JPEG, WEBP, TIFF, BMP, GIF
    """
    account_id = _get_secret("CLOUDFLARE_ACCOUNT_ID")
    api_token = _get_secret("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        raise ValueError(
            "Cloudflare credentials not configured. "
            "Please set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env or Streamlit secrets."
        )

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/"
        "@cf/ocr"
    )
    headers = {
        "Authorization": f"Bearer {api_token}",
    }

    # Encode image as base64
    b64_data = base64.b64encode(file_bytes).decode("utf-8")

    payload = {
        "image": f"data:image/jpeg;base64,{b64_data}",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    result = response.json()
    # Cloudflare OCR returns {"result": {"text": "..."}}
    return result.get("result", {}).get("text", "")


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Auto-detect file type and extract text.

    Supports: PDF (.pdf), images (.png/.jpg/.jpeg/.webp/.tiff/.bmp/.gif).
    """
    ext = os.path.splitext(filename)[-1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    else:
        # Image formats
        return extract_text_from_image(file_bytes)


# ---------------------------------------------------------------------------
# LLM statement analysis
# ---------------------------------------------------------------------------

def analyze_statement(raw_text: str, masked_text: str) -> str:
    """
    Send masked statement text to LLM for explanation.

    Args:
        raw_text: Original extracted text (for debugging/logging — not sent to LLM).
        masked_text: PII-masked version of the statement text.

    Returns:
        Plain-English explanation of the statement.
    """
    try:
        from utils.llm import get_llm
    except ImportError:
        # Fallback if called outside app context
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    prompt = _build_analysis_prompt(masked_text)

    try:
        llm = get_llm()
    except Exception:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    try:
        response = llm.invoke([
            {"role": "system", "content": "You are a helpful healthcare assistant specializing in Singapore's MediShield Life and MediSave scheme."},
            {"role": "user", "content": prompt},
        ])
        return response.content
    except Exception as e:
        return (
            "⚠️ Sorry, something went wrong while analyzing your statement. "
            "Please try again or ensure your file is clearly readable."
        )


def _build_analysis_prompt(masked_text: str) -> str:
    return f"""You are analyzing a masked MediShield Life or MediSave statement.
All personal identifying information has been replaced with [MASKED] labels.

Below is the statement text:
---
{masked_text}
---

Please explain this statement in plain English. Structure your response to cover:

1. **Summary** — What type of document is this (e.g., claim statement, hospital bill, benefit summary)?
2. **What was claimed / charged** — Total bill amount and what it covers
3. **Claim payout breakdown** — How much MediShield Life paid, how much MediSave was used, and any patient responsibility
4. **How the payout was calculated** — Explain if deductible, co-insurance, or pro-ration applied
5. **Shortfalls** — Any amounts not covered by MediShield Life or MediSave, and why
6. **What the patient needs to pay** — Net amount payable by the patient

If certain information is not present in the statement, say so clearly.
Keep your explanation clear and easy to understand for a non-expert.
Format amounts as $X,XXX.
If you cannot determine a value, say "not available in this statement" rather than guessing."""
