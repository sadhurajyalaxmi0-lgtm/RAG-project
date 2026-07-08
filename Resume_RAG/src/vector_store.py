from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document


def create_vector_store(
    chunks: List[Document],
    embeddings,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """Create or load a persistent Chroma vector store from document chunks."""
    base_dir = Path(__file__).resolve().parent.parent
    target_dir = Path(persist_directory).resolve() if persist_directory else base_dir / "chroma_db"

    if target_dir.exists():
        vector_store = Chroma(
            persist_directory=str(target_dir),
            embedding_function=embeddings,
        )
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(target_dir),
        )

    if chunks:
        if target_dir.exists() and not any(target_dir.iterdir()):
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(target_dir),
            )

    print(f"Indexed chunks: {len(chunks)}")
    return vector_store
