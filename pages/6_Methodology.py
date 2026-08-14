"""
Methodology — How the MediShield Life & MediSave Assistant works
"""
import streamlit as st

st.set_page_config(page_title="Methodology", page_icon="🏥")

st.title("How This Assistant Works")

st.markdown("""
## How does it generate answers?

When you ask a question, the assistant:

1. **Searches** a local knowledge base built from official CPF and MOH publications
2. **Finds** the most relevant information from those documents
3. **Generates** a plain-language answer using that information
4. **Cites** the source documents so you can verify the information

The knowledge base is built from publicly available government sources — it does not
connect to CPF's live systems.

## What sources does it use?

All information comes from official Singapore government sources:

- **CPF Board** — MediShield Life and MediSave scheme details
- **Ministry of Health (MOH)** — MediShield Life benefits and claim limits

The knowledge base is updated periodically. For the most current rates and benefits,
always check the official CPF and MOH websites.

## How accurate is it?

We aim for high accuracy, but because the assistant is powered by AI:

- It may occasionally misinterpret or misstate specific amounts
- It cannot guarantee that every answer reflects the very latest policy change
- It should not replace official CPF or MOH publications

**Always verify** important numbers (e.g. specific claim limits, deductibles, premiums)
directly on the [CPF website](https://www.cpf.gov.sg) or [MOH website](https://www.moh.gov.sg).

## How are conversations handled?

The assistant remembers the context of your conversation within a single session.
It does not retain information across sessions, and no personal data is stored.

If you ask a follow-up question, it may reference what you asked before to
give you a more relevant answer.

## What are the assistant's limitations?

| Limitation | What it means |
|---|---|
| No account access | Cannot look up your specific CPF balance, premiums, or claims |
| General information only | Cannot provide personalised financial or medical advice |
| May not have latest rates | Knowledge base is refreshed periodically, not in real time |
| Complex queries | May not handle unusual or complex situations well |

For personalised queries, please contact CPF Board directly.

## Can I use this for official purposes?

No. This tool provides general educational information only. It is not affiliated with
CPF Board, MOH, or the Singapore Government. Nothing on this platform constitutes
official advice or a substitute for official sources.
"""
)

# ── Process Flowcharts ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Process Flowcharts")

# ── Flowchart CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.flowchart-container { display: flex; flex-direction: column; align-items: center; gap: 0; margin: 20px 0; }
.flow-row { display: flex; flex-direction: column; align-items: center; }
.flow-row-horizontal { display: flex; flex-direction: row; align-items: center; justify-content: center; gap: 0; }
.flow-node { padding: 10px 18px; border-radius: 8px; font-size: 13px; font-weight: 500; text-align: center; min-width: 160px; max-width: 220px; }
.flow-node-start { background: #1a4480; color: white; border: 2px solid #1a4480; }
.flow-node-process { background: #c9ab5a; color: #1a1a1a; border: 2px solid #c9ab5a; }
.flow-node-ai { background: #4a7fc1; color: white; border: 2px solid #4a7fc1; }
.flow-node-output { background: #2e7d32; color: white; border: 2px solid #2e7d32; }
.flow-node-stop { background: #b71c1c; color: white; border: 2px solid #b71c1c; }
.flow-node-upload { background: #6a0dad; color: white; border: 2px solid #6a0dad; }
.flow-node-mask { background: #e65100; color: white; border: 2px solid #e65100; }
.flow-node-decision { background: #00695c; color: white; border: 2px solid #00695c; border-radius: 50%; min-width: 40px; max-width: 40px; padding: 10px 14px; }
.arrow { font-size: 20px; color: #555; text-align: center; line-height: 1; }
.arrow-h { font-size: 20px; color: #555; text-align: center; line-height: 1; }
.flow-label { font-size: 11px; color: #666; text-align: center; margin-top: -4px; margin-bottom: 4px; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ── Use Case 1: Chat ──────────────────────────────────────────────────────────
st.markdown("### Use Case 1 — Chat with Information")

with st.container():
    col1, col2 = st.columns([1, 6])
    with col2:
        st.markdown("""
        <div class="flowchart-container">

            <!-- Step 1: User asks question -->
            <div class="flow-row">
                <div class="flow-node flow-node-start">👤 User submits question</div>
            </div>

            <div class="arrow">↓</div>

            <!-- Step 2: Auth gate -->
            <div class="flow-row">
                <div class="flow-node flow-node-process">🔐 Login check (auth gate)</div>
            </div>

            <div class="arrow">↓</div>

            <!-- Step 3: RAG routing / deductible table -->
            <div class="flow-row">
                <div class="flow-node flow-node-decision" title="Classifier">?</div>
            </div>
            <div class="flow-label">Is deductible question?</div>

            <div class="arrow">↓ Yes &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ↓ No</div>

            <!-- Branch: Yes -->
            <div class="flow-row-horizontal">
                <div class="flow-row">
                    <div class="flow-node flow-node-process">📊 Show deductible table</div>
                    <div class="flow-node flow-node-process">💬 Explain in plain English</div>
                </div>

                <div style="font-size:20px; color:#555;">&nbsp;&nbsp;&nbsp;&nbsp;</div>

                <!-- Branch: No -->
                <div class="flow-row">
                    <div class="flow-node flow-node-process">🔍 Vector search (RAG)</div>
                    <div class="flow-node flow-node-ai">🤖 LLM generates answer</div>
                    <div class="flow-node flow-node-output">✅ Display answer + sources</div>
                </div>
            </div>

        </div>
        """, unsafe_allow_html=True)

st.markdown("")
st.markdown("**Use Case 1 — Chat with Information**")
st.caption(
    "The user submits a question about MediShield Life or MediSave. "
    "The system checks login, classifies whether it's a deductible query, and either "
    "displays the deductible table or performs a RAG search followed by LLM answer generation."
)

st.markdown("---")

# ── Use Case 2: Statement Upload ──────────────────────────────────────────────
st.markdown("### Use Case 2 — Statement Upload & Analysis")

with st.container():
    col1, col2 = st.columns([1, 6])
    with col2:
        st.markdown("""
        <div class="flowchart-container">

            <!-- Step 1: User uploads file -->
            <div class="flow-row">
                <div class="flow-node flow-node-upload">📄 Upload statement file</div>
            </div>

            <div class="arrow">↓</div>

            <!-- Step 2: Auth gate -->
            <div class="flow-row">
                <div class="flow-node flow-node-process">🔐 Login check (auth gate)</div>
            </div>

            <div class="arrow">↓</div>

            <!-- Step 3: Format check + OCR extraction -->
            <div class="flow-row">
                <div class="flow-node flow-node-process">🔍 Extract text (OCR for images/scans)</div>
            </div>

            <div class="arrow">↓</div>

            <!-- Step 4: Prompt injection check -->
            <div class="flow-row">
                <div class="flow-node flow-node-decision" title="Decision">🔍 ?</div>
            </div>
            <div class="flow-label">Does the extracted text contain a prompt injection attempt?</div>

            <div class="arrow">↓ No &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ↓ Yes (block)</div>

            <div class="flow-row">
                <div class="flow-node flow-node-stop">🚫 Error: Block upload</div>
            </div>

            <div style="height:4px;"></div>

            <!-- Step 5: PII masking -->
            <div class="flow-row">
                <div class="flow-node flow-node-mask">🎭 Mask PII (NRIC, phone, name, address, bank, DOB)</div>
            </div>

            <div class="arrow">↓</div>

            <!-- Step 6: LLM analysis -->
            <div class="flow-row">
                <div class="flow-node flow-node-ai">🤖 LLM explains statement in plain English</div>
            </div>

            <div class="arrow">↓</div>

            <!-- Step 7: Display result -->
            <div class="flow-row">
                <div class="flow-node flow-node-output">✅ Display explanation + masked text preview</div>
            </div>

        </div>
        """, unsafe_allow_html=True)

st.markdown("")
st.markdown("**Use Case 2 — Statement Upload & Analysis**")
st.caption(
    "The user uploads a MediShield Life or MediSave statement (PDF, DOCX, or image). "
    "The system extracts text via OCR, checks for prompt injection attacks, masks all personal "
    "information (NRIC, phone, name, address, DOB, bank details), sends the masked text to the "
    "LLM for explanation, and displays the plain-English summary alongside a preview of what was sent."
)

st.markdown("---")

# ── Architecture overview ─────────────────────────────────────────────────────
st.markdown("""
## Architecture Overview

The application is built with **Streamlit** and powered by two main backends:

- **RAG Pipeline** (LangChain + OpenAI + FAISS) — for the chat knowledge assistant
- **LLM Analysis Pipeline** — for statement upload explanation

| Component | Technology | Role |
|---|---|---|
| Frontend | Streamlit | Multi-page web UI |
| Auth | bcrypt + session state | Password protection & role-based access |
| RAG Search | LangChain + OpenAI Embeddings + FAISS | Relevant document retrieval |
| LLM | GPT-4o-mini (temperature 0.3) | Answer generation |
| OCR | OCR.space API | Text extraction from images/scans |
| PII Masking | Custom regex patterns | NRIC, phone, name, address, DOB, bank masking |
| Prompt Defence | `_wrap_prompt()` | Neutralises instruction-injection in user input |

All data is stored locally. No personal information is sent to any external database.
""")
