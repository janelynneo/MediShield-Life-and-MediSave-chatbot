"""
MediShield Life & MediSave Chatbot
Streamlit multi-page RAG app.
Run: streamlit run app.py
"""

import os
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

# ── Deductible table ─────────────────────────────────────────────────────────
DEDUCTIBLES = {
    ("Class C", "80 and below"): 2000,
    ("Class C", "81 and above"): 3000,
    ("Class B1", "80 and below"): 2500,
    ("Class B1", "81 and above"): 3500,
    ("Class B2", "80 and below"): 2000,
    ("Class B2", "81 and above"): 3000,
    ("Class A", "80 and below"): 3500,
    ("Class A", "81 and above"): 5000,
    ("Private", "80 and below"): 3500,
    ("Private", "81 and above"): 5000,
    ("Day Surgery", "80 and below"): 1500,
    ("Day Surgery", "81 and above"): 2000,
    ("Outpatient", "80 and below"): 500,
    ("Outpatient", "81 and above"): 1000,
}

WARD_CLASSES = ["Class C", "Class B1", "Class B2", "Class A", "Private", "Day Surgery", "Outpatient"]
AGE_GROUPS = ["80 and below", "81 and above"]

DEDUCTIBLE_TRIGGERS = [
    "deductible", "deductibles", "excess", "what is my deductible",
    "pay before mediashield", "out-of-pocket before",
    "how much do i pay first", "ward deductible",
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
    return any(t in q for t in DEDUCTIBLE_TRIGGERS)


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
    return """You are a helpful assistant specialising in Singapore's MediShield Life health insurance
and MediSave healthcare savings scheme. You answer questions accurately and concisely,
using the official information provided in the context. If you do not know the answer,
say so — do not make up information.

Always cite your sources by mentioning the document title and URL where relevant.
Keep answers in plain English. Use simple language for general audiences.
Format numbers clearly (e.g. $2,000 not 2000).

If a user asks about deductibles, say: "I can help with that! Please select your ward class and age group below." then wait — the app will show the dropdowns.

Relevant official sources:
- CPF MediShield Life: https://www.cpf.gov.sg/member/healthcare-financing/medishield-life
- CPF MediSave: https://www.cpf.gov.sg/member/healthcare-financing/using-your-medisave-savings
- MOH MediShield Life: https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/medishield-life/medishield-life/
- MOH MediSave: https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/medisave/
"""


def rag_answer(question: str, chat_history: list) -> str:
    """Retrieve relevant docs and generate an answer."""
    vectorstore = get_vectorstore()
    llm = get_llm()

    if vectorstore is None:
        return (
            "⚠️ The knowledge base is not set up yet. "
            "Please run `python rag/ingest.py` first to build the search index."
        )

    docs = vectorstore.similarity_search(question, k=5)
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
    if "deductible_step" not in st.session_state:
        st.session_state.deductible_step = None
    if "dw_ward" not in st.session_state:
        st.session_state.dw_ward = None

def reset_deductible():
    st.session_state.deductible_step = None




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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

# ── Deductible Calculator (state machine: pick_ward → pick_age → done) ──────────
step = st.session_state.get("deductible_step", None)

if step == "pick_ward":
    st.markdown("**🧮 MediShield Life Deductible Calculator**")
    st.markdown("Select your ward class or treatment type, then click **Next**.")
    ward = st.selectbox("Ward / Treatment Type", WARD_CLASSES)
    if st.button("Next →", key="dw_next"):
        st.session_state.dw_ward = ward
        st.session_state.deductible_step = "pick_age"
        st.rerun()

elif step == "pick_age":
    st.markdown("**🧮 MediShield Life Deductible Calculator**")
    st.markdown(f"Ward: *{st.session_state.dw_ward}* — now select your age group.")
    age = st.selectbox("Age Group", AGE_GROUPS, key="dw_age")
    if st.button("Calculate Deductible", key="dw_calc"):
        ward = st.session_state.dw_ward
        deductible = DEDUCTIBLES.get((ward, age), 0)
        result = (
            f"**Your MediShield Life deductible: ${deductible:,}**\n\n"
            f"For {ward}, age {age}. This is the amount you pay once per policy year "
            f"before MediShield Life coverage begins.\n\n"
            f"**Annual claim limit:** MediShield Life pays up to $200,000 per policy year "
            f"(there is no lifetime limit).\n\n"
            f"**Co-insurance:** On top of the deductible, you also pay a percentage of the "
            f"remaining bill (co-insurance), up to a cap per policy year."
        )
        st.session_state.chat_history.append(AIMessage(content=result))
        st.session_state.deductible_step = None
        st.rerun()
    if st.button("← Back", key="dw_back"):
        st.session_state.deductible_step = "pick_ward"
        st.rerun()

elif step == "done":
    st.markdown("**🧮 MediShield Life Deductible Calculator**")
    if st.button("💬 Ask something else"):
        reset_deductible()
        st.rerun()


# ── Chat history ─────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ── Chat input ────────────────────────────────────────────────────────────────
question = st.chat_input("Ask anything about MediShield Life or MediSave…", key="chat_input_area")

if question:
    # If deductible conversation is in progress, cancel it
    if st.session_state.deductible_step is not None:
        reset_deductible()

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.chat_history.append(HumanMessage(content=question))

    with st.spinner("Searching official sources…"):
        answer = rag_answer(question, st.session_state.chat_history)

    with st.chat_message("assistant"):
        st.markdown(answer)

        # Auto-trigger deductible calculator only for genuine deductible questions
        if is_deductible_question(question):
            st.session_state.deductible_step = "pick_ward"
            st.rerun()



        # Show sources for regular answers
        vectorstore = get_vectorstore()
        if vectorstore is not None:
            docs = vectorstore.similarity_search(question, k=5)
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
