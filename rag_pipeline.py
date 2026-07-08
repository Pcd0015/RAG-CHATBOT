"""
rag_pipeline.py
----------------
The retrieval-augmented generation core: switched to Google Gemini.
"""
import os
from dataclasses import dataclass, field
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import config

@dataclass
class Source:
    filename: str
    snippet: str
    score: float = 0.0

@dataclass
class RAGResponse:
    answer: str
    sources: List[Source] = field(default_factory=list)

class RAGPipeline:
    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        if not os.path.isdir(config.CHROMA_DIR):
            raise RuntimeError(f"No vector store found at '{config.CHROMA_DIR}'.")

        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.vectordb = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=config.CHROMA_DIR,
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=config.TEMPERATURE,
        )

    def retrieve(self, question: str, k: int = None) -> List[Source]:
        k = k or config.TOP_K
        results = self.vectordb.similarity_search_with_relevance_scores(question, k=k)
        sources = []
        for doc, score in results:
            sources.append(
                Source(
                    filename=doc.metadata.get("source", "unknown"),
                    snippet=doc.page_content,
                    score=round(score, 3),
                )
            )
        return sources

    def _build_context(self, sources: List[Source]) -> str:
        blocks = []
        for i, s in enumerate(sources, 1):
            blocks.append(f"[Chunk {i} | source: {s.filename}]\n{s.snippet}")
        return "\n\n---\n\n".join(blocks)

    def _build_history_messages(self, chat_history: List[dict]):
        msgs = []
        for turn in chat_history or []:
            if turn["role"] == "user":
                msgs.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                msgs.append(AIMessage(content=turn["content"]))
        return msgs

    def query(self, question: str, chat_history: List[dict] = None) -> RAGResponse:
        """
        Full RAG turn with sanitized message sequence for Gemini.
        """
        sources = self.retrieve(question)

        if not sources:
            return RAGResponse(
                answer="I couldn't find anything relevant to answer that.",
                sources=[],
            )

        context = self._build_context(sources)
        user_turn = (
            f"Context from the document collection:\n\n{context}\n\n"
            f"---\n\nQuestion: {question}"
        )

        # Build message sequence: System -> History -> Latest Human Question
        messages = [SystemMessage(content=config.SYSTEM_PROMPT)]
        messages.extend(self._build_history_messages(chat_history))
        messages.append(HumanMessage(content=user_turn))

        # DEBUG: Print the structure before calling the API
        print(f"DEBUG: Sending {len(messages)} messages to Gemini.")
        
        try:
            response = self.llm.invoke(messages)
            return RAGResponse(answer=response.content, sources=sources)
        except Exception as e:
            print(f"CRITICAL ERROR IN LLM INVOKE: {e}")
            raise e