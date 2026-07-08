# RAG Chatbot — Retrieval-Augmented Generation over Your Own Documents

A production-shaped RAG pipeline that answers questions grounded in a private
document collection, with source citations, chat history, a CLI, and a web UI.

**Stack:** Python · LangChain · ChromaDB (vector DB) · Sentence-Transformers
(local embeddings) · Claude (Anthropic API) · Streamlit

---

## Architecture

```
 ┌─────────────┐     ┌────────────────┐     ┌───────────────┐
 │  data/*.pdf │     │  RecursiveChar  │     │  Sentence-     │
 │  *.txt/*.md │ --> │  TextSplitter   │ --> │  Transformers  │
 │ (your docs) │     │  (chunking)     │     │  (embeddings)  │
 └─────────────┘     └────────────────┘     └───────┬────────┘
                                                      │
                                                      v
                                            ┌───────────────────┐
                                            │  ChromaDB          │
                                            │  (persisted vector │
                                            │   store on disk)   │
                                            └─────────┬───────────┘
                                                      │ similarity search (top-K)
                                                      v
 ┌─────────────┐     ┌────────────────┐     ┌───────────────┐
 │ User question│ --> │ Retriever grabs│ --> │  Claude (LLM) │ --> Answer +
 │ (CLI / Web)  │     │ relevant chunks│     │  + context +  │     citations
 └─────────────┘     └────────────────┘     │  chat history │
                                             └───────────────┘
```

**Why this design:**
- **Chunking with overlap** (1000 chars, 150 overlap) preserves context across
  chunk boundaries so answers don't get cut off mid-thought.
- **Local embeddings** (`all-MiniLM-L6-v2`) mean ingestion costs nothing and
  works offline — only the final generation step calls the Claude API.
- **ChromaDB** persists to disk, so you embed once and query many times
  without re-processing documents.
- **Source citations** are returned with every answer (filename + relevance
  score + snippet), so answers are auditable instead of a black box.
- **Grounded system prompt** explicitly instructs the model to say "I don't
  know" rather than hallucinate when the retrieved context is insufficient.

---

## Project Structure

```
rag_chatbot/
├── config.py           # All settings, loaded from .env
├── ingest.py            # Load → chunk → embed → store pipeline
├── rag_pipeline.py       # Retrieval + generation logic (the "RAG" core)
├── cli.py                # Terminal chat interface
├── app.py                 # Streamlit web UI (upload + chat)
├── data/                  # Put your source documents here
│   └── sample_notes.txt   # Example doc so the project works out of the box
├── chroma_db/              # Generated vector store (created after ingest)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your API key
```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```
Get a key at https://console.anthropic.com/

### 3. Add your documents
Drop `.pdf`, `.txt`, or `.md` files into the `data/` folder. A sample file
(`data/sample_notes.txt`) is included so you can test immediately without
adding anything.

### 4. Build the vector index
```bash
python ingest.py
# or, to wipe and rebuild from scratch:
python ingest.py --reset
```
This downloads the local embedding model on first run (~90MB, one-time),
then chunks and embeds every document into `chroma_db/`.

### 5. Chat with your documents

**Terminal:**
```bash
python cli.py
```

**Web UI (recommended):**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. You can upload new documents directly from
the sidebar — they're chunked, embedded, and added to the index live.

---

## Example Session (using the included sample doc)

```
You: What's the API rate limit for enterprise customers?

Assistant: Enterprise tier customers have a rate limit of 5000 requests
per minute, compared to 1000 requests per minute for standard tier
customers. Requests over the limit get an HTTP 429 response with a
Retry-After header. [source: sample_notes.txt]

  Sources: sample_notes.txt
```

Try asking about the rollback window, on-call escalation times, or data
retention policy — all grounded in the actual document, with wrong or
outside-of-doc questions correctly flagged as "not found."

---

## Customization Guide

| Want to... | Change this |
|---|---|
| Use a bigger/cheaper Claude model | `ANTHROPIC_MODEL` in `.env` |
| Support more file types (e.g. .docx, .html) | Add a loader in `ingest.py`'s `loaders` list (LangChain has 100+ built-in loaders) |
| Retrieve more/fewer chunks per answer | `TOP_K` in `.env` |
| Larger/smaller chunks | `CHUNK_SIZE` / `CHUNK_OVERLAP` in `.env` |
| Swap ChromaDB for Pinecone/FAISS | Replace the `Chroma(...)` calls in `ingest.py` and `rag_pipeline.py` — LangChain's vector store interface is consistent across providers |
| Use OpenAI instead of Claude | Swap `ChatAnthropic` for `ChatOpenAI` in `rag_pipeline.py` |
| Re-rank retrieved chunks for higher precision | Add a cross-encoder re-ranking step after `retrieve()` in `rag_pipeline.py` |

---

## What This Project Demonstrates

- **Embeddings & semantic search** — converting text to vectors and finding
  meaning-based matches rather than keyword matches.
- **Document chunking strategy** — balancing chunk size/overlap to preserve
  context without diluting retrieval precision.
- **Vector database usage** — persistent storage, similarity search with
  relevance scoring, incremental re-indexing.
- **LLM orchestration** — prompt engineering for groundedness, multi-turn
  conversation state, citation tracking.
- **End-to-end product thinking** — not just a script, but a CLI, a web UI,
  configuration management, and a re-ingestion workflow a real user could
  operate.

---

## Known Limitations (worth mentioning if you present this project)

- Similarity search alone can miss answers that require reasoning across
  many documents at once (no multi-hop retrieval).
- No re-ranking step, so the top-K chunks are used as-is; a cross-encoder
  re-ranker would improve precision on large corpora.
- ChromaDB here runs locally/embedded — for large scale or multi-user
  production use, a managed vector DB (Pinecone, Weaviate Cloud, pgvector)
  would be a more realistic next step.
