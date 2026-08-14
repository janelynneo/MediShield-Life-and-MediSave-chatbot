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

There are two ways to use this assistant:

**Chat with the FAQ assistant**
Simply type your question in the chat box. For example:

- *"What does MediShield Life cover?"*
- *"How much can I withdraw from MediSave for my hospital stay?"*
- *"What is the deductible for a Class B1 ward?"*
- *"Can I use MediSave to pay for my parent's medical bills?"*

The assistant searches a local knowledge base of official CPF and MOH publications,
then generates an answer using that information — citing the source documents
so you can verify everything.

**Upload a statement for explanation**
If you have a MediShield Life or MediSave statement (PDF, DOCX, or image), you
can upload it on the **Statement Upload** page. The assistant will:

1. Extract the text using OCR
2. Mask any personal information (NRIC, phone number, name, address, etc.)
3. Explain the statement in plain English

Your personal information is never sent to any external service.

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

Your question is first matched against a local knowledge base of official CPF and
MOH publications. If relevant documents are found, their content is combined with
your question and sent to OpenAI to generate an answer — your CPF account
information is never accessed or sent. No personal data is retained after the
session ends.
"""
)
