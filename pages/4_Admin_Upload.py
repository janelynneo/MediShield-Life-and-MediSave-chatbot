"""
Admin Upload page — for admins to manage the RAG knowledge base.

Admin can upload official documents (PDF, TXT, MD) to extend the
knowledge base. Documents are saved to disk and the FAISS index is
updated incrementally.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure .env is loaded for local dev
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

from utils.auth import verify_user, get_role

# ── Auth — admin only ───────────────────────────────────────────────────────────
def require_admin():
    if not st.session_state.get("is_logged_in", False):
        st.error("Please log in to access this page.")
        st.stop()
    if st.session_state.get("role") != "admin":
        st.error("You do not have permission to access this page.")
        st.stop()

require_admin()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Admin Upload — MediShield Assistant",
    page_icon="🔧",
    layout="centered",
)

st.title("🔧 Admin: Document Management")
st.caption("Upload official documents to extend the RAG knowledge base.")

# ── Paths ──────────────────────────────────────────────────────────────────────
UPLOAD_DIR = BASE_DIR / "data" / "sample_docs" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_document(file_bytes: bytes, filename: str):
    """
    Parse a document and return LangChain Document objects.
    Supports: PDF (PyPDF2), TXT, MD (TextLoader).
    """
    from langchain_core.documents import Document
    from langchain_community.document_loaders import TextLoader
    from PyPDF2 import PdfReader
    import io

    ext = os.path.splitext(filename)[-1].lower()
    docs = []

    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            all_text = "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            if all_text.strip():
                docs.append(Document(page_content=all_text, metadata={
                    "source_key": filename,
                    "title": filename.replace("_", " ").replace(".pdf", "").title(),
                    "url": "https://www.cpf.gov.sg",
                    "type": "admin_upload",
                }))
        except Exception as e:
            return [], f"Failed to parse PDF: {e}"

    elif ext in (".txt", ".md"):
        try:
            # Write to temp file for TextLoader
            tmp_path = UPLOAD_DIR / f"_tmp_{filename}"
            with open(tmp_path, "wb") as f:
                f.write(file_bytes)
            loader = TextLoader(str(tmp_path), encoding="utf-8")
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source_key"] = filename
                doc.metadata["title"] = filename.replace("_", " ").replace(".md", "").replace(".txt", "").title()
                doc.metadata["url"] = "https://www.cpf.gov.sg"
                doc.metadata["type"] = "admin_upload"
                docs.append(doc)
            # Clean up tmp file
            tmp_path.unlink(missing_ok=True)
        except Exception as e:
            return [], f"Failed to parse file: {e}"

    else:
        return [], f"Unsupported file type: {ext}"

    return docs, None


def rebuild_index():
    """Rebuild the entire FAISS index from base docs + uploaded docs."""
    import json
    from langchain_community.document_loaders import TextLoader
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from utils.rag import INDEX_DIR

    DOCS_DIR = BASE_DIR / "data" / "sample_docs"
    MANIFEST_PATH = UPLOAD_DIR / ".upload_manifest.json"

    all_docs = []

    # Load base documents
    base_files = list(DOCS_DIR.glob("*.md"))
    for f in base_files:
        try:
            loader = TextLoader(str(f), encoding="utf-8")
            for doc in loader.load():
                doc.metadata["source_key"] = f.stem
                doc.metadata["title"] = f.stem.replace("_", " ").title()
                doc.metadata["type"] = "base"
                all_docs.append(doc)
        except Exception:
            pass

    # Load uploaded documents
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r") as f:
                manifest = json.load(f)
            for entry in manifest:
                fpath = Path(entry["path"])
                if fpath.exists():
                    try:
                        loader = TextLoader(str(fpath), encoding="utf-8")
                        for doc in loader.load():
                            doc.metadata["source_key"] = entry["filename"]
                            doc.metadata["title"] = entry["filename"].replace("_", " ").replace(".pdf", "").replace(".txt", "").replace(".md", "").title()
                            doc.metadata["url"] = "https://www.cpf.gov.sg"
                            doc.metadata["type"] = "admin_upload"
                            all_docs.append(doc)
                    except Exception:
                        pass
        except Exception:
            pass

    if not all_docs:
        return False, "No documents found to index."

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    chunks = splitter.split_documents(all_docs)

    # Build index
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    return True, f"Index rebuilt with {len(chunks)} chunks from {len(all_docs)} documents."


# ── Upload widget ──────────────────────────────────────────────────────────────
st.markdown("### 📤 Upload Document")

uploaded_file = st.file_uploader(
    "Upload a PDF, TXT, or MD file to add to the knowledge base",
    type=["pdf", "txt", "md"],
)

if uploaded_file:
    col_upload, col_cancel = st.columns([1, 1])
    with col_upload:
        upload_clicked = st.button("⬆️ Upload & Index", use_container_width=True)
    with col_cancel:
        if st.button("Clear", use_container_width=True):
            st.rerun()

    if upload_clicked:
        import datetime, json as json_lib

        filename = uploaded_file.name
        file_bytes = uploaded_file.read()
        file_size = len(file_bytes)

        # Save file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_name = f"{timestamp}_{filename}"
        saved_path = UPLOAD_DIR / saved_name
        with open(saved_path, "wb") as f:
            f.write(file_bytes)

        # Parse document
        docs, parse_err = parse_document(file_bytes, filename)
        if parse_err:
            st.error(f"❌ {parse_err}")
            saved_path.unlink(missing_ok=True)
            st.stop()

        # Chunk documents
        from utils.rag import chunk_documents, add_to_index, add_to_manifest

        chunks = chunk_documents(docs)
        success = add_to_index(chunks, source_name=filename)
        if success:
            add_to_manifest(filename, str(saved_path), len(chunks), file_size)
            st.success(f"✅ Uploaded and indexed: **{filename}** ({len(chunks)} chunks added)")
        else:
            st.error("❌ Failed to update the index. Please try again.")
            saved_path.unlink(missing_ok=True)
        st.rerun()

# ── Document manifest ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📚 Indexed Documents")

from utils.rag import get_upload_manifest

manifest = get_upload_manifest()

if manifest:
    import datetime
    rows = []
    for entry in manifest:
        added = entry.get("added_at", "Unknown")
        try:
            dt = datetime.datetime.fromisoformat(added)
            added = dt.strftime("%d %b %Y, %H:%M")
        except Exception:
            pass
        rows.append({
            "Filename": entry.get("filename", "Unknown"),
            "Chunks": entry.get("chunk_count", "?"),
            "Size (KB)": round(entry.get("file_size_bytes", 0) / 1024, 1),
            "Added": added,
            "Type": entry.get("path", "").endswith(".md") and "base" or "uploaded",
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.info("No uploaded documents yet. Upload a file above to get started.")

# ── Rebuild index ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔄 Rebuild Index")

st.warning(
    "⚠️ **Full rebuild** — re-indexes ALL base documents plus all uploaded files. "
    "Use this if the index is stale or corrupted."
)

col_rebuild, _ = st.columns([1, 3])
with col_rebuild:
    if st.button("🔄 Rebuild Full Index", use_container_width=True):
        with st.spinner("Rebuilding index… this may take a moment…"):
            ok, msg = rebuild_index()
            if ok:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")
        st.rerun()
