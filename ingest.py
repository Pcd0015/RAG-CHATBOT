"""
ingest.py
---------
Loads every supported document from DATA_DIR, splits them into overlapping
chunks, embeds each chunk with a local sentence-transformers model, and
persists everything into a Chroma vector database on disk.
"""
import argparse
import os
import sys
import time

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config


def load_documents(data_dir: str):
    """Load .pdf, .txt, and .md files from data_dir using the right loader for each type."""
    if not os.path.isdir(data_dir) or not os.listdir(data_dir):
        print(f"No files found in '{data_dir}'. Add documents there and re-run.")
        sys.exit(1)

    docs = []
    loaders = [
        (".pdf", PyPDFLoader),
        (".txt", TextLoader),
        (".md", UnstructuredMarkdownLoader),
    ]

    for ext, loader_cls in loaders:
        loader = DirectoryLoader(
            data_dir,
            glob=f"**/*{ext}",
            loader_cls=loader_cls,
            show_progress=False,
            use_multithreading=True,
        )
        try:
            loaded = loader.load()
            docs.extend(loaded)
            if loaded:
                print(f"  Loaded {len(loaded)} {ext} document(s)")
        except Exception as e:
            print(f"  Warning: failed loading {ext} files: {e}")

    if not docs:
        print("No documents could be loaded. Check file formats in DATA_DIR.")
        sys.exit(1)

    return docs


def chunk_documents(docs):
    """Split raw documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Ensure every chunk carries a clean "source" filename for citations later.
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        chunk.metadata["source"] = os.path.basename(src)

    return chunks


def build_vector_store(chunks, reset: bool = False):
    """Embed chunks and persist them into a local Chroma collection."""
    # FIX: Added model_kwargs={'device': 'cpu'} to resolve NotImplementedError
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    if reset and os.path.isdir(config.CHROMA_DIR):
        import shutil
        shutil.rmtree(config.CHROMA_DIR)
        print(f"Cleared existing vector store at '{config.CHROMA_DIR}'")

    vectordb = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DIR,
    )

    # Batch adds to avoid overwhelming memory on very large corpora
    batch_size = 100
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        vectordb.add_documents(batch)
        print(f"  Embedded {min(i + batch_size, total)}/{total} chunks")

    return vectordb


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument("--reset", action="store_true", help="Wipe and rebuild the vector store")
    parser.add_argument("--data-dir", default=config.DATA_DIR, help="Directory containing source documents")
    args = parser.parse_args()

    start = time.time()

    print(f"Loading documents from '{args.data_dir}' ...")
    docs = load_documents(args.data_dir)
    print(f"Loaded {len(docs)} document(s) total.")

    print("Chunking documents ...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks (chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")

    print("Embedding and storing in ChromaDB (first run downloads the embedding model) ...")
    build_vector_store(chunks, reset=args.reset)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Vector store persisted at '{config.CHROMA_DIR}'.")
    print("You can now run: python cli.py   OR   streamlit run app.py")


if __name__ == "__main__":
    main()