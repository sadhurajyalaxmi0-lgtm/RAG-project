from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader


def load_resumes(resume_dir: Optional[str] = None) -> List[Document]:
    """Load all PDF resumes from the specified directory as LangChain documents."""
    base_dir = Path(__file__).resolve().parent.parent
    target_dir = Path(resume_dir).resolve() if resume_dir else base_dir / "data" / "resumes"

    if not target_dir.exists():
        raise FileNotFoundError(f"Resume directory not found: {target_dir}")

    if not target_dir.is_dir():
        raise NotADirectoryError(f"Resume path is not a directory: {target_dir}")

    pdf_files = sorted(target_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF resumes found in {target_dir}")
        return []

    try:
        loader = PyPDFDirectoryLoader(str(target_dir))
        documents = loader.load()
    except Exception as exc:
        raise RuntimeError(f"Failed to load resumes from {target_dir}: {exc}") from exc

    print(f"Loaded {len(pdf_files)} resumes from {target_dir}")
    return documents
