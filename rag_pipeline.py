"""
rag_pipeline.py
----------------
The retrieval-augmented generation core with strict message sequence sanitization.
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
        sources = [Source(filename=d.metadata.get("source", "unknown"), snippet=d.page_content, score=round(s, 3)) for d, s in results]
        return sources

    def _build_context(self, sources: List[Source]) -> str:
        return "\n\n---\n\n".join([f"[Chunk {i} | source: {s.filename}]\n{s.snippet}" for i, s in enumerate(sources, 1)])

    def _build_history_messages(self, chat_history: List[dict]):
        """Sanitizes history to ensure strict alternating User/AI sequence."""
        msgs = []
        last_role = None
        for turn in chat_history or []:
            current_role = turn["role"]
            if current_role == last_role: continue # Skip if sequence is broken
            
            if current_role == "user":
                msgs.append(HumanMessage(content=turn["content"]))
                last_role = "user"
            elif current_role == "assistant":
                msgs.append(AIMessage(content=turn["content"]))
                last_role = "assistant"
        return msgs

    def query(self, question: str, chat_history: List[dict] = None) -> RAGResponse:
        sources = self.retrieve(question)
        if not sources:
            return RAGResponse(answer="I couldn't find relevant info.", sources=[])

        context = self._build_context(sources)
        user_turn = f"Context from documents:\n\n{context}\n\n---\n\nQuestion: {question}"

        # Build message sequence: System -> Filtered History -> Latest User Message
        messages = [SystemMessage(content=config.SYSTEM_PROMPT)]
        messages.extend(self._build_history_messages(chat_history))
        
        # Ensure the very last message is always a HumanMessage
        if not messages or not isinstance(messages[-1], HumanMessage):
            messages.append(HumanMessage(content=user_turn))
        else:
            # If the last message was somehow already a HumanMessage, just append content
            messages[-1].content += f"\n\n{user_turn}"

        try:
            response = self.llm.invoke(messages)
            return RAGResponse(answer=response.content, sources=sources)
        except Exception as e:
            print(f"CRITICAL GEMINI ERROR: {e}")
            raise e