import os

import streamlit as st

from src import config
from src.ingest import build_index
from src.rag_chain import build_rag_chain, answer_question

st.set_page_config(page_title="PolicyPal", page_icon="📄")
st.title("📄 PolicyPal — Insurance Policy Q&A Assistant")
st.caption("Ask plain-English questions about a policy document. Answers are grounded and sourced.")

with st.sidebar:
    st.header("1. Add your policy document(s)")
    uploaded_files = st.file_uploader(
        "Upload policy PDFs or .txt files", type=["txt", "pdf"], accept_multiple_files=True
    )

    if st.button("Build index", type="primary"):
        if not uploaded_files:
            st.warning("Upload at least one file first.")
        else:
            os.makedirs(config.DOCS_DIR, exist_ok=True)
            for f in uploaded_files:
                with open(os.path.join(config.DOCS_DIR, f.name), "wb") as out:
                    out.write(f.getbuffer())
            with st.spinner("Chunking, embedding, and indexing ..."):
                build_index(config.DOCS_DIR, config.INDEX_DIR)
            st.success("Index built. You can start asking questions.")

if "chain" not in st.session_state:
    st.session_state.chain = None

if os.path.exists(config.INDEX_DIR) and st.session_state.chain is None:
    with st.spinner("Loading existing index and model ..."):
        st.session_state.chain, st.session_state.vectorstore = build_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("e.g. Is dental treatment covered?")

if question:
    if st.session_state.chain is None:
        st.error("Build the index first (see sidebar).")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking ..."):
                answer, sources = answer_question(st.session_state.chain, st.session_state.vectorstore, question)
            st.markdown(answer)
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**{s['source']}**")
                    st.caption(s["snippet"] + " ...")
        st.session_state.messages.append({"role": "assistant", "content": answer})