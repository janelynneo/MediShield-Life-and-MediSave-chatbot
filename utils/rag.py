"""
RAG helpers — vector store lifecycle, index path.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

BASE_DIR = Path(__file__).parent.parent
INDEX_DIR = BASE_DIR / "rag" / "index"


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
