"""
MediShield Life & MediSave Chatbot
Streamlit multi-page RAG app.
Run: streamlit run app.py
"""

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "rag" / "index"

# ── Env ──────────────────────────────────────────────────────────────────────
load_dotenv(BASE_DIR / ".env")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediShield Life & MediSave Assistant",
    page_icon="🏥",
    layout="centered",
)

# ── All configurable limits (loaded from data/limits.csv) ───────────────────────
import csv, os

_LIMITS_CSV = BASE_DIR / "data" / "limits.csv"
os.makedirs(BASE_DIR / "data", exist_ok=True)

# Structure: LIMITS["section"][key] = value
# DEDUCTIBLES[(ward, age)] = deductible  (from deductible section)
# TOSP[tosp_code] = {"medishield": n, "medisave": n}
# RADIOTHERAPY[name] = {"medishield": n, "medisave": n}
# OTHER_TREATMENTS[name] = {"medishield": n, "medisave": n}
# PRORATION_INPATIENT[(ward, citizenship)] = "XX%"
# PRORATION_DAYSURG[(setting, citizenship)] = "XX%"
# PRORATION_CH[(setting, citizenship)] = "XX%"
LIMITS: dict = {}
DEDUCTIBLES: dict = {}
TOSP: dict = {}
RADIOTHERAPY: dict = {}
OTHER_TREATMENTS: dict = {}
PRORATION_INPATIENT: dict = {}
PRORATION_DAYSURG: dict = {}
PRORATION_CH: dict = {}

def _int(val):
    try:
        return int(str(val).strip().replace(",", "").replace("%", ""))
    except (ValueError, AttributeError):
        return None

def _pct(val):
    s = str(val).strip()
    return s if s else None

if _LIMITS_CSV.exists():
    with open(_LIMITS_CSV) as f:
        for row in csv.DictReader(f):
            sec = row.get("section", "").strip()
            cat = row.get("category", "").strip()
            item = row.get("item", "").strip()
            ms = _int(row.get("medishield_life", ""))
            mv = _int(row.get("medisave", ""))

            # Section header row
            if sec.startswith("[") and sec.endswith("]"):
                continue

            # Section is in the section column (lowercase)
            current = sec.lower()

            if current == "limits":
                if item:
                    LIMITS.setdefault("main_limits", {})[item] = ms

            elif current == "deductible":
                ward = cat
                age = item
                if ward and age and ms is not None:
                    DEDUCTIBLES[(ward, age)] = ms

            elif current == "tosp table":
                code = cat   # "1A", "2B" etc.
                if code:
                    TOSP[code] = {"medishield": ms, "medisave": mv}

            elif current == "radiotherapy":
                name = cat
                if name:
                    RADIOTHERAPY[name] = {"medishield": ms, "medisave": mv}

            elif current == "other treatments":
                name = cat
                if name:
                    OTHER_TREATMENTS[name] = {"medishield": ms, "medisave": mv}

            elif current == "maximum claim limit":
                name = cat
                if name:
                    LIMITS.setdefault("max_claim", {})[name] = ms

            elif current == "proration_inpatient":
                ward = cat
                citizenship = item
                if ward and citizenship:
                    PRORATION_INPATIENT[(ward, citizenship)] = _pct(row.get("medishield_life",""))

            elif current == "proration_daysurg":
                setting = cat
                citizenship = item
                if setting and citizenship:
                    PRORATION_DAYSURG[(setting, citizenship)] = _pct(row.get("medishield_life",""))

            elif current == "proration_community_hospital" or current == "proration community hospital":
                setting = cat
                citizenship = item
                if setting and citizenship:
                    PRORATION_CH[(setting, citizenship)] = _pct(row.get("medishield_life",""))

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

WARD_CLASSES = ["Class C", "Class B1", "Class B2", "Class A", "Private", "Day Surgery", "Outpatient"]
AGE_GROUPS = ["80 and below", "81 and above"]

def _fmt(n): return f"${n:,}"

DEDUCTIBLE_TRIGGERS = [
    r"\bdeductible\b", r"\bdeductibles\b", r"\bexcess\b",
    r"\bward deductible\b", r"\bout-of-pocket\b",
    r"\bpay before mediashield\b",
]


# ── LLM & vector store (session-state singleton) ─────────────────────────────
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


@st.cache_resource
def get_vectorstore():
    if not INDEX_DIR.exists():
        return None
    return FAISS.load_local(
        str(INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def is_deductible_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(t, q) for t in DEDUCTIBLE_TRIGGERS)


def format_docs(docs: list[Document]) -> str:
    """Concatenate docs with source citations."""
    parts = []
    for d in docs:
        title = d.metadata.get("title", d.metadata.get("source_key", "Unknown"))
        url = d.metadata.get("url", "")
        source_link = f"**[{title}]({url})**" if url else f"**{title}**"
        parts.append(f"[Source: {source_link}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def build_system_prompt() -> str:
    ml = LIMITS.get("main_limits", {})
    icu_ms      = _fmt(ml.get("icu_limit_per_day", 5140))
    icu_first2  = _fmt(ml.get("first_2_days", 800))
    ms_first2   = _fmt(ml.get("first_2_days", 1130))   # same for hosp
    ms_day3     = _fmt(ml.get("day3_onwards", 400))
    annual      = _fmt(LIMITS.get("max_claim", {}).get("annual_limit", 200000))
    return f"""You are a helpful assistant specialising in Singapore's MediShield Life health insurance
and MediSave healthcare savings scheme. You answer questions accurately and concisely,
using the official information provided in the context. If you do not know the answer,
say so — do not make up information.

Always cite your sources by mentioning the document title and URL where relevant.
Keep answers in plain English. Use simple language for general audiences.
Format numbers clearly (e.g. $2,000 not 2000).

**IMPORTANT — Only use values from the context below.** If the exact dollar amount is NOT
in the retrieved context, say "I don't have that specific information" instead of guessing.
NEVER invent a number. Never return a figure like "$2,000" unless it explicitly appears
in the context.

**Distinguish MediShield Life from MediSave:**
- MediShield Life = insurance plan. ICU: {icu_ms}/day + {icu_first2}/day first 2 days.
- MediSave = savings account. For MediSave ICU and hospitalization: use the same withdrawal limits —
  first 2 days {ms_first2}/day, day 3 onwards {ms_day3}/day. Return these exact figures when asked
  about MediSave ICU or hospitalization limits.

If a user asks about deductibles, say: "I can help with that! Please select your ward class and age group below." then wait — the app will show the dropdowns.

Relevant official sources:
- CPF MediShield Life: https://www.cpf.gov.sg/member/healthcare-financing/medishield-life
- CPF MediSave: https://www.cpf.gov.sg/member/healthcare-financing/using-your-medisave-savings
- MOH MediShield Life: https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/medishield-life/medishield-life/
- MOH MediSave: https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/medisave/
"""


def get_relevant_docs(question: str, k: int = 5) -> list:
    """Retrieve docs, boosting MediSave docs for MediSave queries."""
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return []
    docs = vectorstore.similarity_search(question, k=k)
    # Boost: if question mentions MediSave-specific terms, upweight medisave docs
    q_lower = question.lower()
    medisave_terms = ["medisave", "icu", "hospitalisation", "hospitalization", "withdraw", "balance", "savings", "hospital bill", "daily ward"]
    shield_terms = ["medishield", "premium", "insured", "claim limit", "co-insurance", "coinsurance", "deductible"]
    is_medisave = any(t in q_lower for t in medisave_terms) and not any(t in q_lower for t in shield_terms)
    if is_medisave:
        medisave_docs = [d for d in docs if "medisave" in d.metadata.get("source_key", "").lower()]
        other_docs = [d for d in docs if "medisave" not in d.metadata.get("source_key", "").lower()]
        docs = medisave_docs + other_docs
    return docs[:k]


def rag_answer(question: str, chat_history: list) -> str:
    """Retrieve relevant docs and generate an answer."""
    vectorstore = get_vectorstore()
    llm = get_llm()

    if vectorstore is None:
        return (
            "⚠️ The knowledge base is not set up yet. "
            "Please run `python rag/ingest.py` first to build the search index."
        )

    docs = get_relevant_docs(question, k=5)
    context = format_docs(docs)

    history_text = ""
    if chat_history:
        history_lines = []
        for msg in chat_history[-6:]:
            if isinstance(msg, HumanMessage):
                history_lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_lines.append(f"Assistant: {msg.content}")
        history_text = "\n".join(history_lines)

    if history_text:
        system_and_history = (
            f"{build_system_prompt()}\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Relevant information:\n{context}"
        )
    else:
        system_and_history = (
            f"{build_system_prompt()}\n\n"
            f"Relevant information:\n{context}"
        )

    response = llm.invoke([
        {"role": "system", "content": system_and_history},
        {"role": "user", "content": question},
    ])
    return response.content


# ── Conversation state helpers ────────────────────────────────────────────────
def init_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None




# ── UI ─────────────────────────────────────────────────────────────────────
st.title("🏥 MediShield Life & MediSave Assistant")
st.caption(
    "Ask about MediShield Life, MediSave, premiums, claim limits, coverage, and more. "
    "Based on official CPF and MOH sources."
)

init_state()

if get_vectorstore() is None:
    st.warning(
        "⚠️ The search index is not loaded. "
        "Make sure you've run `python rag/ingest.py` first."
    )

# ── Clear chat ───────────────────────────────────────────────────────────────
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# ── Suggested questions ─────────────────────────────────────────────────────
st.markdown("### Try asking")
SAMPLE_QUERIES = [
    "What does MediShield Life cover?",
    "How much can I withdraw from MediSave?",
    "What are the MediShield Life claim limits?",
    "Can I use MediSave for my family?",
    "How do I pay MediShield Life premiums?",
]
cols = st.columns(3)
for i, q in enumerate(SAMPLE_QUERIES):
    with cols[i % 3]:
        if st.button(q, use_container_width=True):
            st.session_state.chat_input_area = q

# ── Chat history ─────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

# ── Chat input ────────────────────────────────────────────────────────────────
question = st.chat_input("Ask anything about MediShield Life or MediSave…", key="chat_input_area")

if question:
    st.session_state.pending_question = question

if st.session_state.get("pending_question"):
    with st.chat_message("user"):
        st.markdown(st.session_state.pending_question)

    st.session_state.chat_history.append(HumanMessage(content=st.session_state.pending_question))

    with st.spinner("Searching official sources…"):
        answer = rag_answer(st.session_state.pending_question, st.session_state.chat_history)

    with st.chat_message("assistant"):
        st.markdown(answer)

        # Show deductible calculator if this was a deductible question
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
                    result = (
                        f"**Deductible: ${deductible:,}** "
                        f"(ward class **{ward_sel}**, age **{age_sel}**)\n\n"
                        f"The deductible is the amount you pay once per policy year before MediShield Life coverage begins.\n\n"
                        f"**Annual claim limit:** Up to {_fmt(LIMITS.get('max_claim', {}).get('annual_limit', 200000))} per policy year (no lifetime limit).\n"
                        f"**Co-insurance:** You also pay 10-20% of the remaining bill."
                    )
                    st.session_state.chat_history.append(AIMessage(content=result))
                    st.session_state.pending_question = None
                    st.rerun()
            with c2:
                if st.button("Skip calculator"):
                    st.session_state.pending_question = None
                    st.rerun()

        # Show sources for regular answers
        if get_vectorstore() is not None:
            docs = get_relevant_docs(st.session_state.pending_question, k=5)
            if docs:
                sources = {(d.metadata.get("title", ""), d.metadata.get("url", "")) for d in docs}
                with st.expander("📄 Sources"):
                    for title, url in sorted(sources):
                        st.markdown(f"- [{title}]({url})")

    st.session_state.chat_history.append(AIMessage(content=answer))

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<div style="display:flex; align-items:center; gap:12px; margin-top:8px;">'
    f'<img src="file://{BASE_DIR}/static/cpf_logo.png" width="110" '
    f'style="pointer-events:none; display:block;"/>'
    f'<div>'
    f'<div style="font-size:0.85em; color:#555;">'
    f'Not medical advice. For authoritative guidance, visit '
    f'<a href="https://www.cpf.gov.sg" target="_blank">CPF</a> or '
    f'<a href="https://www.moh.gov.sg" target="_blank">MOH</a>.'
    f'</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)
