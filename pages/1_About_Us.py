"""
About the MediShield Life & MediSave Assistant
"""
import streamlit as st

st.set_page_config(page_title="About Us", page_icon="🏥")

st.title("About this Assistant")

st.markdown("""
## What is this tool?

The **MediShield Life & MediSave Assistant** is a conversational AI tool that helps
members of the public in Singapore understand their MediShield Life health insurance
and MediSave healthcare savings scheme.

It answers common questions about:
- What MediShield Life and MediSave cover
- Deductibles, co-insurance, and claim limits
- MediSave withdrawal limits and eligible uses
- Premiums and subsidies
- How to use MediSave for yourself and your family

## Who is this for?

Anyone who wants quick, clear answers about their MediShield Life and MediSave benefits
— without having to navigate lengthy government websites or wait on a helpline.

## How do I use it?

Simply type your question in the chat box. For example:

- *"What does MediShield Life cover?"*
- *"How much can I withdraw from MediSave for my hospital stay?"*
- *"What is the deductible for a Class B1 ward?"*
- *"Can I use MediSave to pay for my parent's medical bills?"*

The assistant will search official CPF and MOH sources and give you an answer,
citing where the information comes from.

## What it cannot do

This assistant:

- **Does not access your personal information** — it has no connection to your CPF account
- **Does not provide financial advice** — it gives general information only
- **Does not make claims decisions** — only the CPF Board can process actual claims
- **Does not have the most current rates** — always verify important amounts on the
  [CPF website](https://www.cpf.gov.sg) or [MOH website](https://www.moh.gov.sg)

For queries beyond this tool's scope, please refer to the official
[CPF website](https://www.cpf.gov.sg) or [MOH website](https://www.moh.gov.sg) for guidance.

## Data and privacy

Your question is sent to OpenAI to generate an answer. No personal CPF account
information is accessed or stored. No data is retained after the session ends.
"""
)
