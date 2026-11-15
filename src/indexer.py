"""Step 2 - Embeddings and vector index (FAISS)."""
import os
import site
from pathlib import Path

# Ensure Windows loads faiss DLL before langchain loads it
if hasattr(os, "add_dll_directory"):
    try:
        for sp in site.getsitepackages():
            faiss_dir = Path(sp) / "faiss"
            if faiss_dir.is_dir():
                os.add_dll_directory(str(faiss_dir))
    except Exception:
        pass
try:
    import faiss
except Exception:
    pass

from . import config

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


def get_embeddings(offline: bool = False) -> Embeddings:
    if offline:
        from .offline import HashEmbeddings
        return HashEmbeddings()
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def _index_path(offline: bool) -> str:
    return str(Path(config.INDEX_DIR) / ("faiss_offline" if offline else "faiss"))


def build_index(docs, offline: bool = False) -> FAISS:
    docs = list(docs)
    if not docs:
        raise ValueError("Cannot build an index without documents.")
    store = FAISS.from_documents(docs, get_embeddings(offline))
    Path(config.INDEX_DIR).mkdir(parents=True, exist_ok=True)
    store.save_local(_index_path(offline))
    return store


def load_index(offline: bool = False) -> FAISS:
    path = _index_path(offline)
    if not Path(path).exists():
        # Fall back to offline index if available
        if not offline and Path(_index_path(True)).exists():
            return load_index(offline=True)
        raise FileNotFoundError("No index yet. Run:  python app.py build")
    # The index is created by this project on your own machine, so loading it is safe.
    try:
        return FAISS.load_local(path, get_embeddings(offline), allow_dangerous_deserialization=True)
    except Exception:
        # A stale online index may require a model that is not installed. Prefer
        # a valid local fallback so opening the app does not trigger a download.
        offline_path = _index_path(True)
        if not offline and Path(offline_path).exists():
            return FAISS.load_local(
                offline_path, get_embeddings(offline=True), allow_dangerous_deserialization=True
            )
        raise
