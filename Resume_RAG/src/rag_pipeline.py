from typing import Any, Dict

from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

from .prompt import SYSTEM_PROMPT
from .utils import LOGGER


def build_rag_pipeline(
    vector_store: Chroma,
    model_name: str = "llama3.2",
    top_k: int = 3,
) -> RetrievalQA:
    """Create a RetrievalQA chain for answering questions from retrieved resume chunks."""
    llm = Ollama(model=model_name, temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    prompt = PromptTemplate(
        input_variables=["question"],
        template=f"{SYSTEM_PROMPT}\n\nQuestion: {{question}}",
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def query_resume(
    vector_store: Chroma,
    question: str,
    model_name: str = "llama3.2",
    top_k: int = 3,
) -> Dict[str, Any]:
    """Query the resume RAG system and return the answer with source context."""
    if not question or not question.strip():
        return {"answer": "NOT EXIST", "source_files": [], "retrieved_chunks": []}

    try:
        qa_chain = build_rag_pipeline(vector_store, model_name=model_name, top_k=top_k)
        result = qa_chain.invoke(question)
    except Exception as exc:
        LOGGER.exception("RAG query failed: %s", exc)
        return {"answer": "NOT EXIST", "source_files": [], "retrieved_chunks": []}

    source_documents = result.get("source_documents", [])
    source_files = [doc.metadata.get("source", "unknown") for doc in source_documents]
    answer = str(result.get("result", "")).strip()

    if not answer:
        answer = "NOT EXIST"

    return {
        "answer": answer,
        "source_files": source_files,
        "retrieved_chunks": [doc.page_content for doc in source_documents],
    }
