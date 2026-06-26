"""
Optional Streamlit UI for Wiki-RAG.

Run with: streamlit run streamlit_app.py

Useful for recording a demo GIF/screenshot for the README — not
required for the core pipeline to work (the CLI in app.py is the
primary interface).
"""

import streamlit as st

from src.config import Config
from src.pipeline import RAGPipeline

st.set_page_config(page_title="Wiki-RAG", page_icon="🔎")
st.title("🔎 Wiki-RAG")
st.caption("Local Retrieval-Augmented Generation over Wikipedia ML/AI articles, powered by Ollama.")


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(Config())


pipeline = get_pipeline()

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Setup")
    if st.button("Build / rebuild index"):
        with st.spinner("Fetching articles and building vector index..."):
            pipeline.build_index()
        st.success("Index built.")
    st.markdown("---")
    st.markdown(
        "Make sure Ollama is running locally with:\n\n"
        "`ollama pull llama3.2:3b`\n\n"
        "`ollama pull nomic-embed-text`"
    )

query = st.text_input("Ask a question about ML / AI:")

if st.button("Ask") and query:
    with st.spinner("Retrieving context and generating answer..."):
        result = pipeline.ask(query)
    st.session_state.history.insert(0, (query, result))

for q, result in st.session_state.history:
    st.markdown(f"**Q: {q}**")
    st.write(result.answer)
    with st.expander("Sources"):
        for src in result.sources:
            st.markdown(f"- [{src.source_title}]({src.source_url}) (distance={src.distance:.3f})")
    st.markdown("---")
