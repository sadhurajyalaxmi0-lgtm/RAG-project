from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .utils import LOGGER


def create_vector_store(
    chunks: List[Document],
    embeddings,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """Create or load a persistent Chroma vector store from document chunks."""
    base_dir = Path(__file__).resolve().parent.parent
    target_dir = Path(persist_directory).resolve() if persist_directory else base_dir / "chroma_db"

    try:
        if target_dir.exists() and any(target_dir.iterdir()):
            LOGGER.info("Loading existing ChromaDB from %s", target_dir)
            vector_store = Chroma(
                persist_directory=str(target_dir),
                embedding_function=embeddings,
            )
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            if chunks:
                LOGGER.info("Creating new ChromaDB at %s", target_dir)
                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=str(target_dir),
                )
            else:
                LOGGER.warning("No chunks available; creating empty ChromaDB at %s", target_dir)
                vector_store = Chroma(
                    persist_directory=str(target_dir),
                    embedding_function=embeddings,
                )
    except Exception as exc:
        LOGGER.exception("Failed to initialize ChromaDB: %s", exc)
        raise RuntimeError(f"Failed to initialize ChromaDB: {exc}") from exc

    LOGGER.info("Indexed chunks: %s", len(chunks))
    return vector_store
