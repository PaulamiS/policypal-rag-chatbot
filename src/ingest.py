import argparse
import os

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src import config


def load_documents(docs_dir: str):
    txt_loader = DirectoryLoader(
        docs_dir, glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    pdf_loader = DirectoryLoader(
        docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    documents = txt_loader.load() + pdf_loader.load()
    if not documents:
        raise ValueError(f"No .txt or .pdf files found in {docs_dir}.")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_index(docs_dir: str, index_dir: str):
    print(f"Loading documents from {docs_dir} ...")
    documents = load_documents(docs_dir)
    print(f"Loaded {len(documents)} document(s). Splitting into chunks ...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunk(s). Embedding ...")

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"FAISS index saved to {index_dir}")
    return vectorstore


def load_index(index_dir: str):
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    return FAISS.load_local(
        index_dir, embeddings, allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", default=config.DOCS_DIR)
    parser.add_argument("--index_dir", default=config.INDEX_DIR)
    args = parser.parse_args()
    build_index(args.docs_dir, args.index_dir)