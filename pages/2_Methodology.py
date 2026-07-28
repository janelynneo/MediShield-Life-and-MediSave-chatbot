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
