"""
cli.py
------
Minimal terminal chat interface for the RAG pipeline. Good for quick testing
before wiring up the Streamlit UI.

Usage:
    python cli.py
"""
from rag_pipeline import RAGPipeline


def main():
    print("=" * 60)
    print("RAG Chatbot (CLI) - type 'exit' or 'quit' to leave, 'reset' to clear history")
    print("=" * 60)

    try:
        pipeline = RAGPipeline()
    except RuntimeError as e:
        print(f"\nSetup error: {e}")
        return

    history = []

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if question.lower() == "reset":
            history = []
            print("(conversation history cleared)")
            continue

        result = pipeline.query(question, chat_history=history)

        print(f"\nAssistant: {result.answer}")
        if result.sources:
            unique_files = sorted({s.filename for s in result.sources})
            print(f"\n  Sources: {', '.join(unique_files)}")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result.answer})


if __name__ == "__main__":
    main()
