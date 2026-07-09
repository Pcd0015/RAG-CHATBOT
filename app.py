import os
import time
import streamlit as st
import config
from ingest import load_documents, chunk_documents, build_vector_store

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

# --- AUTO-INGESTION LOGIC ---
if not os.path.exists(config.CHROMA_PATH):
    st.info("Vector store not found. Initializing ingestion...")
    if os.path.exists(config.DATA_DIR) and len(os.listdir(config.DATA_DIR)) > 0:
        with st.spinner("Building vector index..."):
            docs = load_documents(config.DATA_DIR)
            chunks = chunk_documents(docs)
            build_vector_store(chunks)
        st.success("Vector index built!")
    else:
        st.warning(f"No documents found in {config.DATA_DIR}.")

def get_pipeline():
    from rag_pipeline import RAGPipeline
    if "pipeline" not in st.session_state:
        try:
            st.session_state.pipeline = RAGPipeline()
        except Exception as e:
            st.session_state.pipeline = None
            st.session_state.pipeline_error = str(e)
    return st.session_state.get("pipeline")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📁 Document Collection")
    uploaded_files = st.file_uploader("Upload files", type=["pdf", "txt", "md"], accept_multiple_files=True)

    if uploaded_files and st.button("Add to knowledge base"):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        for f in uploaded_files:
            path = os.path.join(config.DATA_DIR, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
        with st.spinner("Processing..."):
            docs = load_documents(config.DATA_DIR)
            chunks = chunk_documents(docs)
            build_vector_store(chunks)
        st.rerun()

    st.divider()
    if os.path.isdir(config.DATA_DIR):
        files = os.listdir(config.DATA_DIR)
        st.caption(f"**{len(files)} file(s)** currently indexed:")
        for f in files: st.caption(f"• {f}")

# --- MAIN CHAT AREA ---
st.title("📚 RAG Chatbot")
pipeline = get_pipeline()

if pipeline is None:
    st.warning(f"Error: {st.session_state.get('pipeline_error', 'System not initialized.')}")
    st.stop()

if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]: st.markdown(f"**{s.filename}** (score: {s.score:.2f})")

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant"):
        start_time = time.time()
        
        # Generator function for streaming
        def response_generator():
            result = pipeline.query(question, chat_history=history)
            # Store full result for later use
            st.session_state.temp_result = result
            yield result.answer

        # Stream the response
        full_response = st.write_stream(response_generator())
        
        # Display sources
        result = st.session_state.temp_result
        if result.sources:
            with st.expander("📎 Sources"):
                for s in result.sources:
                    st.markdown(f"**{s.filename}** (score: {s.score:.2f})")
        
        # Log latency
        st.sidebar.caption(f"⏱️ Last response: {time.time() - start_time:.2f}s")

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "sources": result.sources}
    )