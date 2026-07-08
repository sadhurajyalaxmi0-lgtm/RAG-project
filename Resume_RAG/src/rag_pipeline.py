from typing import Any, Dict, List, Optional

from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .prompt import SYSTEM_PROMPT


def build_rag_pipeline(
    vector_store: Chroma,
    model_name: str = "llama3.2",
    top_k: int = 3,
) -> RetrievalQA:
    """Create a RetrievalQA chain for answering questions from retrieved resume chunks."""
    llm = Ollama(model=model_name, temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": None},
    )


def query_resume(
    vector_store: Chroma,
    question: str,
    model_name: str = "llama3.2",
    top_k: int = 3,
) -> Dict[str, Any]:
    """Query the resume RAG system and return the answer with source context."""
    qa_chain = build_rag_pipeline(vector_store, model_name=model_name, top_k=top_k)
    result = qa_chain.invoke(question)

    source_documents = result.get("source_documents", [])
    source_files = []
    for doc in source_documents:
        source_files.append(doc.metadata.get("source", "unknown"))

    return {
        "answer": result.get("result", ""),
        "source_files": source_files,
        "retrieved_chunks": [doc.page_content for doc in source_documents],
    }
