import logging
from pathlib import Path


def setup_logging() -> logging.Logger:
    """Configure and return a module logger."""
    logger = logging.getLogger("resume_rag")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


LOGGER = setup_logging()


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent
