"""
Build the FAISS vector index from MediShield Life and MediSave source documents.
Run: python rag/ingest.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "data" / "sample_docs"
INDEX_DIR = BASE_DIR / "rag" / "index"

load_dotenv()


def main():
    if not DOCS_DIR.exists():
        print(f"Error: Documents directory not found: {DOCS_DIR}")
        print("Please ensure source documents are in data/sample_docs/")
        return

    doc_files = list(DOCS_DIR.glob("*.md"))
    if not doc_files:
        print(f"Error: No .md files found in {DOCS_DIR}")
        return

    print(f"Loading {len(doc_files)} document(s)...")

    DOC_URLS = {
        "medishield_life": "https://www.cpf.gov.sg/member/healthcare-financing/medishield-life",
        "medisave": "https://www.cpf.gov.sg/member/healthcare-financing/using-your-medisave-savings",
        "medishield_life_booklet": "https://www.cpf.gov.sg/content/dam/web/member/healthcare/documents/InformationBookletForTheNewlyInsured.pdf",
    }

    docs = []
    for f in doc_files:
        print(f"  Loading: {f.name}")
        loader = TextLoader(f, encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source_key"] = f.stem
            doc.metadata["title"] = f.stem.replace("_", " ").title()
            doc.metadata["url"] = DOC_URLS.get(f.stem, "https://www.cpf.gov.sg")
        docs.extend(loaded)

    print(f"Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "## ", "### ", "| ", "  ", ". "],
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("Generating embeddings (this may take a moment)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"Saved index to: {INDEX_DIR}")
    print("Done! You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
