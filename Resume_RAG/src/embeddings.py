from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Return a local Hugging Face embedding model instance."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)
