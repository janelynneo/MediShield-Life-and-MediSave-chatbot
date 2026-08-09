"""
RAG helpers — vector store lifecycle, index path, incremental indexing.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).parent.parent
INDEX_DIR = BASE_DIR / "rag" / "index"
UPLOAD_DIR = BASE_DIR / "data" / "sample_docs" / "uploads"
MANIFEST_PATH = UPLOAD_DIR / ".upload_manifest.json"


def get_vectorstore():
    """Load the persisted FAISS index (cached after first load)."""
    if not INDEX_DIR.exists():
        return None
    # Import here to avoid circular dependency at module level
    from utils.llm import get_embeddings
    return FAISS.load_local(
        str(INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def chunk_documents(docs: list[Document], chunk_size: int = 400, chunk_overlap: int = 80) -> list[Document]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", "## ", "### ", "|", "  ", ". "],
    )
    return splitter.split_documents(docs)


def add_to_index(documents: list[Document], source_name: str = "upload") -> bool:
    """
    Add new documents to the existing FAISS index incrementally.

    Args:
        documents: LangChain Document objects to add.
        source_name: Human-readable name for this document set (used in metadata).

    Returns:
        True on success, False on failure.
    """
    try:
        from utils.llm import get_embeddings

        vectorstore = get_vectorstore()
        if vectorstore is None:
            # No existing index — create a new one
            embeddings = get_embeddings()
            vectorstore = FAISS.from_documents(documents, embeddings)
        else:
            # Add to existing index
            vectorstore.add_documents(documents)

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(INDEX_DIR))
        return True
    except Exception:
        return False


def get_upload_manifest() -> list[dict]:
    """Load the upload manifest listing all uploaded files."""
    import json
    if not MANIFEST_PATH.exists():
        return []
    try:
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


def add_to_manifest(filename: str, file_path: str, doc_count: int, file_size: int) -> None:
    """Add a record to the upload manifest."""
    import json
    import datetime
    manifest = get_upload_manifest()
    manifest.append({
        "filename": filename,
        "path": str(file_path),
        "added_at": datetime.datetime.now().isoformat(),
        "chunk_count": doc_count,
        "file_size_bytes": file_size,
    })
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
