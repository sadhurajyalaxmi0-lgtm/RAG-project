from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader

from .utils import LOGGER


def load_resumes(resume_dir: Optional[str] = None) -> List[Document]:
    """Load all PDF resumes from the specified directory as LangChain documents."""
    base_dir = Path(__file__).resolve().parent.parent
    target_dir = Path(resume_dir).resolve() if resume_dir else base_dir / "data" / "resumes"

    if not target_dir.exists():
        LOGGER.error("Resume directory not found: %s", target_dir)
        raise FileNotFoundError(f"Resume directory not found: {target_dir}")

    if not target_dir.is_dir():
        LOGGER.error("Resume path is not a directory: %s", target_dir)
        raise NotADirectoryError(f"Resume path is not a directory: {target_dir}")

    pdf_files = sorted(target_dir.glob("*.pdf"))
    if not pdf_files:
        LOGGER.warning("No PDF resumes found in %s", target_dir)
        return []

    documents: List[Document] = []

    try:
        loader = PyPDFDirectoryLoader(str(target_dir))
        documents = loader.load()
    except Exception as exc:
        LOGGER.warning("Directory loader failed for %s: %s", target_dir, exc)

    if not documents:
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                documents.extend(loader.load())
                LOGGER.info("Loaded PDF: %s", pdf_file.name)
            except Exception as inner_exc:
                LOGGER.warning("Skipping invalid PDF %s: %s", pdf_file.name, inner_exc)

    LOGGER.info("Loaded %s resumes from %s", len(documents), target_dir)
    return documents
