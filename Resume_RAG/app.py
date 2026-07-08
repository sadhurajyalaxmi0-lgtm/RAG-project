from pathlib import Path

import streamlit as st

from src.chunking import chunk_documents
from src.data_loader import load_resumes
from src.embeddings import get_embeddings_model
from src.rag_pipeline import query_resume
from src.utils import LOGGER
from src.vector_store import create_vector_store


st.set_page_config(page_title="Resume RAG Assistant", page_icon="📄")
st.title("Resume RAG Assistant")

st.info("This app uses the resumes stored in the local data/resumes folder.")

resume_dir = Path(__file__).resolve().parent / "data" / "resumes"

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if st.session_state.vector_store is None:
    try:
        documents = load_resumes(str(resume_dir))
        if not documents:
            st.warning("No valid PDF resumes were found in the resumes folder.")
            LOGGER.warning("No valid resumes loaded from %s", resume_dir)
        else:
            chunks = chunk_documents(documents)
            embeddings = get_embeddings_model()
            st.session_state.vector_store = create_vector_store(chunks, embeddings)
    except Exception as exc:
        LOGGER.exception("Failed to initialize resume index: %s", exc)
        st.error(f"Failed to initialize resume index: {exc}")

question = st.text_input("Ask a question")

if st.button("Submit") and question.strip():
    try:
        if st.session_state.vector_store is None:
            result = {"answer": "NOT EXIST", "source_files": [], "retrieved_chunks": []}
        else:
            result = query_resume(st.session_state.vector_store, question)

        st.subheader("Answer")
        st.write(result.get("answer", "NOT EXIST"))

        st.subheader("Retrieved resumes")
        source_files = result.get("source_files", [])
        if source_files:
            for source_file in source_files:
                st.write(f"- {source_file}")
        else:
            st.write("No resumes retrieved")

        st.subheader("Retrieved chunks")
        retrieved_chunks = result.get("retrieved_chunks", [])
        if retrieved_chunks:
            for idx, chunk in enumerate(retrieved_chunks, start=1):
                st.write(f"{idx}. {chunk}")
        else:
            st.write("No chunks retrieved")
    except Exception as exc:
        LOGGER.exception("Failed to answer question: %s", exc)
        st.error(f"Failed to answer question: {exc}")
