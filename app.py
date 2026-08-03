"""
MediShield Life & MediSave Chatbot — Streamlit entry point.
All logic lives in utils/llm.py and utils/rag.py; this file handles UI only.
Run: streamlit run app.py
"""

import csv
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from utils.llm import (
    get_llm,
    get_vectorstore,
    rag_answer,
    is_deductible_question,
    is_coverage_compare_question,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# ── Env ──────────────────────────────────────────────────────────────────────
load_dotenv(BASE_DIR / ".env")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediShield Life & MediSave Assistant",
    page_icon="🏥",
    layout="centered",
)

# ── All configurable limits (loaded from data/limits.csv) ────────────────────────
_LIMITS_CSV = BASE_DIR / "data" / "limits.csv"
os.makedirs(BASE_DIR / "data", exist_ok=True)
_csv_loaded = False

# Structure: LIMITS["section"][key] = value
# DEDUCTIBLES[(ward, age)] = deductible
# TOSP[code]          = {"medishield": n, "medisave": n}
# RADIOTHERAPY[name]  = {"medishield": n, "medisave": n}
# OTHER_TREATMENTS[name] = {"medishield": n, "medisave": n}
# PRORATION_INPATIENT[(ward, citizenship)] = "XX%"
# PRORATION_DAYSURG[(setting, citizenship)] = "XX%"
# PRORATION_CH[(setting, citizenship)] = "XX%"
# COINSURANCE = [(bracket, pct), ...]
LIMITS: dict = {}
DEDUCTIBLES: dict = {}
TOSP: dict = {}
RADIOTHERAPY: dict = {}
OTHER_TREATMENTS: dict = {}
PRORATION_INPATIENT: dict = {}
PRORATION_DAYSURG: dict = {}
PRORATION_CH: dict = {}
COINSURANCE: list = []

WARD_CLASSES = ["Class C", "Class B1", "Class B2", "Class A", "Private", "Day Surgery", "Outpatient"]
AGE_GROUPS = ["80 and below", "81 and above"]

# ── Curated FAQs (based on official CPF/MOH sources) ──────────────────────────
FAQS = [
    "How do I make a MediShield Life claim?",
    "Can I use MediSave for my parents' medical bills?",
    "What is the deductible and when does it apply?",
    "Can I opt out of MediShield Life?",
    "What is co-insurance and how does it work?",
    "What is the difference between MediShield Life and MediSave?",
    "How much can I withdraw from MediSave for hospitalization?",
    "What does the annual claim limit mean?",
    "How is my MediShield Life premium determined?",
    "Can I use MediSave for outpatient treatments?",
    "What happens at age 65 with my MediShield Life coverage?",
    "Can I use MediSave to pay for my child's medical bills?",
]


def _int(val):
    try:
        return int(str(val).strip().replace(",", "").replace("%", ""))
    except (ValueError, AttributeError):
        return None


def _pct(val):
    s = str(val).strip()
    return s if s else None


def _fmt(n):
    return f"${n:,}"


if _LIMITS_CSV.exists():
    with open(_LIMITS_CSV) as f:
        for row in csv.DictReader(f):
            sec = row.get("section", "").strip().lower()
            cat = row.get("category", "").strip()
            item = row.get("item", "").strip()
            ms = _int(row.get("medishield_life", ""))
            mv = _int(row.get("medisave", ""))

            if sec.startswith("[") and sec.endswith("]"):
                continue

            if sec == "limits":
                if item:
                    LIMITS.setdefault("main_limits", {})[item] = ms

            elif sec == "deductible":
                if cat and item and ms is not None:
                    DEDUCTIBLES[(cat, item)] = ms

            elif sec == "tosp table":
                if cat:
                    TOSP[cat] = {"medishield": ms, "medisave": mv}

            elif sec == "radiotherapy":
                if cat:
                    RADIOTHERAPY[cat] = {"medishield": ms, "medisave": mv}

            elif sec == "other treatments":
                if cat:
                    OTHER_TREATMENTS[cat] = {"medishield": ms, "medisave": mv}

            elif sec == "maximum claim limit":
                if cat:
                    LIMITS.setdefault("max_claim", {})[cat] = ms

            elif sec == "proration_inpatient":
                if cat and item:
                    PRORATION_INPATIENT[(cat, item)] = _pct(row.get("medishield_life", ""))

            elif sec == "proration_daysurg":
                if cat and item:
                    PRORATION_DAYSURG[(cat, item)] = _pct(row.get("medishield_life", ""))

            elif sec in ("proration_community_hospital", "proration community hospital"):
                if cat and item:
                    PRORATION_CH[(cat, item)] = _pct(row.get("medishield_life", ""))

            elif sec == "co-insurance" and cat and ms is not None:
                COINSURANCE.append((cat, ms))
    _csv_loaded = True
else:
    LIMITS["main_limits"] = {
        "ward_limit_per_day": 830,
        "icu_limit_per_day": 5140,
        "first_2_days": 800,
        "day_surgery_per_day": 830,
        "psychiatric_per_day": 230,
        "psych_first_2_days": 1400,
        "ch_rehab_per_day": 370,
        "ch_subacute_per_day": 570,
    }
    LIMITS["max_claim"] = {"annual_limit": 200000}
    DEDUCTIBLES = {
        ("Class C", "80 and below"): 2000,
        ("Class C", "81 and above"): 2750,
        ("Class B1", "80 and below"): 2500,
        ("Class B1", "81 and above"): 3500,
        ("Class B2", "80 and below"): 2000,
        ("Class B2", "81 and above"): 3500,
        ("Class A", "80 and below"): 3500,
        ("Class A", "81 and above"): 4500,
        ("Private", "80 and below"): 3500,
        ("Private", "81 and above"): 4500,
        ("Day Surgery", "80 and below"): 1500,
        ("Day Surgery", "81 and above"): 2000,
        ("Outpatient", "80 and below"): 500,
        ("Outpatient", "81 and above"): 1000,
    }


# ── Web scraper — fetch official CPF PDF and update CSV ──────────────────────
def _fetch_official_limits():
    """Download the official CPF InfoBooklet PDF and return (text, error_msg)."""
    try:
        import urllib.request, ssl, io
        from pdfminer.high_level import extract_text
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        url = "https://www.cpf.gov.sg/content/dam/web/member/healthcare/documents/InformationBookletForTheNewlyInsured.pdf"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as resp:
            pdf_bytes = resp.read()
        text = extract_text(io.BytesIO(pdf_bytes))
        return text, None
    except Exception as exc:
        return None, f"Failed to fetch CPF PDF: {exc}"


# ── Conversation state ────────────────────────────────────────────────────────
def init_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🏥 MediShield Life & MediSave Assistant")
st.caption(
    "Ask about MediShield Life, MediSave, premiums, claim limits, coverage, and more. "
    "Based on official CPF and MOH sources."
)

init_state()

# Sidebar: refresh from official CPF PDF
with st.sidebar:
    st.markdown("### ⚙️ Data")
    if st.button("📥 Refresh from CPF official site"):
        with st.spinner("Downloading official CPF InfoBooklet PDF…"):
            text, err = _fetch_official_limits()
            if err:
                st.error(f"❌ {err}")
            elif text:
                st.success("✅ Official PDF scraped successfully!")
                ded_idx = text.find("6.2 What is the deductible")
                if ded_idx > 0:
                    st.text_area("Deductible section preview", text[ded_idx:ded_idx + 500],
                                 height=120, disabled=True)

    st.markdown("### ❓ Frequently Asked Questions")
    with st.expander("See all FAQs", expanded=False):
        for q in FAQS:
            if st.button(q, use_container_width=True, key=f"faq_{q}"):
                st.session_state.chat_input_area = q

if get_vectorstore() is None:
    st.warning(
        "⚠️ The search index is not set up yet. "
        "Make sure you've run `python rag/ingest.py` first."
    )

if _LIMITS_CSV.exists() and not _csv_loaded:
    st.warning("⚠️ The limits CSV was found but could not be parsed. Using default values.")

# Clear chat
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Suggested questions
st.markdown("### Try asking")
SAMPLE_QUERIES = [
    "What does MediShield Life cover?",
    "How much can I withdraw from MediSave?",
    "What are the MediShield Life claim limits?",
    "Can I use MediSave for my family?",
    "How do I pay MediShield Life premiums?",
    "Compare MediShield Life vs MediSave coverage",
]
cols = st.columns(3)
for i, q in enumerate(SAMPLE_QUERIES):
    with cols[i % 3]:
        if st.button(q, use_container_width=True):
            st.session_state.chat_input_area = q

# Chat history
for msg in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

# Chat input
question = st.chat_input("Ask anything about MediShield Life or MediSave…", key="chat_input_area")

if question:
    st.session_state.pending_question = question

if st.session_state.get("pending_question"):
    with st.chat_message("user"):
        st.markdown(st.session_state.pending_question)

    st.session_state.chat_history.append(
        HumanMessage(content=st.session_state.pending_question)
    )

    with st.spinner("Searching official sources…"):
        try:
            answer = rag_answer(
                st.session_state.pending_question,
                st.session_state.chat_history,
                LIMITS,
            )
        except Exception as exc:
            print(f"[rag_answer] Unexpected error: {exc}", exc)
            answer = (
                "⚠️ Sorry, something went wrong while generating your answer. "
                "Please try again in a moment."
            )

    with st.chat_message("assistant"):
        st.markdown(answer)

        # ── Deductible calculator ──────────────────────────────────────────
        if is_deductible_question(st.session_state.pending_question):
            st.markdown("---")
            st.markdown("**🧮 MediShield Life Deductible Calculator**")
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox("Ward / Treatment Type", WARD_CLASSES, key="calc_ward")
            with col2:
                st.selectbox("Age Group", AGE_GROUPS, key="calc_age")
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("🧮 Calculate Deductible"):
                    ward_sel = st.session_state.get("calc_ward", WARD_CLASSES[0])
                    age_sel = st.session_state.get("calc_age", AGE_GROUPS[0])
                    deductible = DEDUCTIBLES.get((ward_sel, age_sel), 0)
                    annual_limit_raw = LIMITS.get("max_claim", {}).get("annual_limit", 200000)
                    result = (
                        f"**Deductible: {_fmt(deductible)}** "
                        f"(ward class **{ward_sel}**, age **{age_sel}**)\n\n"
                        f"The deductible is the fixed amount you pay once per policy year "
                        f"before MediShield Life starts paying.\n\n"
                        f"**Annual claim limit:** Up to {_fmt(annual_limit_raw)} per policy year.\n\n"
                        f"**Co-insurance (on remaining bill after deductible):**\n"
                        f"  • First $5,000: 10%\n"
                        f"  • Next $5,000: 5%\n"
                        f"  • Above $10,000: 3%\n\n"
                        f"The claim is subject to your deductible and co-insurance. "
                        f"MediShield Life covers the remaining amount, up to claim limits."
                    )
                    st.session_state.chat_history.append(AIMessage(content=result))
                    st.session_state.pending_question = None
                    st.rerun()
            with c2:
                if st.button("Skip calculator"):
                    st.session_state.pending_question = None
                    st.rerun()

        # ── Coverage comparison ────────────────────────────────────────────
        if is_coverage_compare_question(st.session_state.pending_question):
            st.markdown("---")
            st.markdown("**⚖️ MediShield Life vs MediSave Coverage Comparison**")
            ml = LIMITS.get("main_limits", {})
            annual = _fmt(LIMITS.get("max_claim", {}).get("annual_limit", 200000))

            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🛡️ MediShield Life")
                    st.markdown(f"**Daily ward limit:** {_fmt(ml.get('ward_limit_per_day', 830))}/day")
                    st.markdown(f"**ICU:** {_fmt(ml.get('icu_limit_per_day', 5140))}/day")
                    st.markdown(f"**First 2 days (extra):** +{_fmt(ml.get('first_2_days', 800))}/day")
                    st.markdown(f"**Day Surgery:** {_fmt(ml.get('day_surgery_per_day', 830))}/day")
                    st.markdown(f"**Psychiatric:** {_fmt(ml.get('psychiatric_per_day', 230))}/day")
                    st.markdown(f"**Annual claim limit:** {annual}")
                    st.markdown("**Purpose:** Insurance against large hospital bills")
                    st.markdown("**Who it's for:** All Singapore citizens & PRs (automatic)")

                with col2:
                    st.markdown("### 💰 MediSave")
                    st.markdown(f"**Hospitalisation (first 2 days):** {_fmt(ml.get('first_2_days', 1130))}/day")
                    st.markdown(f"**Hospitalisation (day 3+):** {_fmt(ml.get('day3_onwards', 400))}/day")
                    st.markdown(f"**ICU (first 2 days):** {_fmt(ml.get('first_2_days', 1130))}/day")
                    st.markdown(f"**ICU (day 3+):** {_fmt(ml.get('day3_onwards', 400))}/day")
                    st.markdown(f"**Day Surgery:** {_fmt(ml.get('day_surgery_per_day', 830))}/day")
                    st.markdown("**Purpose:** Pay for hospitalisation & selected outpatient treatments")
                    st.markdown("**Who it's for:** CPF members with accumulated savings")

            st.markdown("---")
            st.markdown("**Key differences:**")
            st.markdown("| | MediShield Life | MediSave |")
            st.markdown("|---|---:|---:|")
            st.markdown(f"| Type | Insurance (premiums payable) | Savings (your CPF money) |")
            st.markdown(f"| Purpose | Protect against large bills | Pay hospitalisation costs |")
            st.markdown(
                f"| Daily ward limit | {_fmt(ml.get('ward_limit_per_day', 830))} "
                f"| {_fmt(ml.get('first_2_days', 1130))} first 2 days, "
                f"{_fmt(ml.get('day3_onwards', 400))} after |"
            )
            st.markdown(f"| Annual limit | {annual} | No limit (uses your balance) |")
            st.markdown(f"| Citizenship | Automatic for SC/PR | Must have CPF savings |")

        # ── Sources ────────────────────────────────────────────────────────
        if get_vectorstore() is not None:
            from utils.llm import get_relevant_docs
            docs = get_relevant_docs(st.session_state.pending_question, k=5)
            if docs:
                sources = {(d.metadata.get("title", ""), d.metadata.get("url", ""))
                           for d in docs}
                with st.expander("📄 Sources"):
                    for title, url in sorted(sources):
                        st.markdown(f"- [{title}]({url})")

    st.session_state.chat_history.append(AIMessage(content=answer))

# Footer
st.markdown("---")
LOGO_PATH = BASE_DIR / "static" / "cpf_logo.png"
footer_left = (
    f'<img src="file://{LOGO_PATH}" width="110" style="pointer-events:none; display:block;"/>'
    if LOGO_PATH.exists()
    else '<span style="font-size:0.85em; color:#555;">🏥 MediShield Life &amp; MediSave</span>'
)
st.markdown(
    f'<div style="display:flex; align-items:center; gap:12px; margin-top:8px;">'
    f'{footer_left}'
    f'<div>'
    f'<div style="font-size:0.85em; color:#555;">'
    f'Not medical advice. For authoritative guidance, visit '
    f'<a href="https://www.cpf.gov.sg" target="_blank">CPF</a> or '
    f'<a href="https://www.moh.gov.sg" target="_blank">MOH</a>. '
    f'Questions or feedback? '
    f'<a href="https://www.cpf.gov.sg/service/write-to-us" target="_blank">Write to us</a>.'
    f'</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)
