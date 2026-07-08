import os
import streamlit as st
import config
from ingest import load_documents, chunk_documents, build_vector_store

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

# --- AUTO-INGESTION LOGIC (Self-Healing on Cloud) ---
if not os.path.exists(config.CHROMA_PATH):
    st.info("Vector store not found. Initializing ingestion of data/ directory...")
    if os.path.exists(config.DATA_DIR) and len(os.listdir(config.DATA_DIR)) > 0:
        with st.spinner("Building vector index for the first time..."):
            docs = load_documents(config.DATA_DIR)
            chunks = chunk_documents(docs)
            build_vector_store(chunks)
        st.success("Vector index built successfully!")
    else:
        st.warning(f"No documents found in {config.DATA_DIR}. Please upload files in the sidebar.")

def get_pipeline():
    from rag_pipeline import RAGPipeline
    if "pipeline" not in st.session_state:
        try:
            st.session_state.pipeline = RAGPipeline()
        except RuntimeError as e:
            st.session_state.pipeline = None
            st.session_state.pipeline_error = str(e)
    return st.session_state.get("pipeline")

# ---------------- Sidebar: document management ----------------
with st.sidebar:
    st.header("📁 Document Collection")

    uploaded_files = st.file_uploader(
        "Upload .pdf, .txt, or .md files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Add to knowledge base"):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        for f in uploaded_files:
            path = os.path.join(config.DATA_DIR, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
        with st.spinner("Chunking and embedding documents..."):
            docs = load_documents(config.DATA_DIR)
            chunks = chunk_documents(docs)
            build_vector_store(chunks)
        st.success(f"Ingested {len(uploaded_files)} file(s). Reloading...")
        st.session_state.pop("pipeline", None)
        st.rerun()

    st.divider()
    if os.path.isdir(config.DATA_DIR):
        files = os.listdir(config.DATA_DIR)
        st.caption(f"**{len(files)} file(s)** currently indexed:")
        for f in files:
            st.caption(f"• {f}")

    st.divider()
    st.caption(f"Model: `{config.GEMINI_MODEL}`")
    st.caption(f"Embeddings: `{config.EMBEDDING_MODEL}`")

    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Main chat area ----------------
st.title("📚 RAG Chatbot")
pipeline = get_pipeline()

if pipeline is None:
    st.warning(f"**Not ready:** {st.session_state.get('pipeline_error', 'Waiting for API Key...')}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    # Note: Depending on your Source dataclass structure, 
                    # adjust s.score to be accessible
                    st.markdown(f"**{s.filename}** (score: {s.score:.2f})")

# Handle new user input
if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            # Result is a RAGResponse object
            result = pipeline.query(question, chat_history=history)
            
            # Explicitly display only the answer string
            st.markdown(result.answer)
            
            # Display sources if available
            if result.sources:
                with st.expander("📎 Sources"):
                    for s in result.sources:
                        st.markdown(f"**{s.filename}** (score: {s.score:.2f})")

    # Store assistant response as string (result.answer) not the object
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )