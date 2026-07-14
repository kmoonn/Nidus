"""CLI entry point for Nidus RAG — supports index, ask, and interactive modes."""

import argparse
import sys

from .chunker import chunk_documents
from .config import get_config
from .embedder import Embedder
from .loader import load_directory
from .pipeline import Pipeline
from .store import Store


def cmd_index(args):
    """Build the document index from PDF files in docs/."""
    config = get_config()

    print("=" * 60)
    print("  Nidus RAG — Index Builder")
    print("=" * 60)

    # Step 1: Load documents
    print("\n[1/4] Loading PDF documents...")
    documents = load_directory(config.docs_dir)
    if not documents:
        print("No documents loaded. Exiting.")
        sys.exit(1)
    print(f"  → {len(documents)} pages loaded")

    # Step 2: Chunk documents
    print("\n[2/4] Chunking documents...")
    chunker_cfg = config.chunker
    chunks = chunk_documents(
        documents,
        chunk_size=chunker_cfg["chunk_size"],
        chunk_overlap=chunker_cfg["chunk_overlap"],
    )
    print(f"  → {len(chunks)} chunks created (size={chunker_cfg['chunk_size']}, overlap={chunker_cfg['chunk_overlap']})")

    # Step 3: Generate embeddings
    print("\n[3/4] Generating embeddings...")
    embedder = Embedder()
    batch_size = 50
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        embeddings = embedder.embed_texts(texts)
        all_embeddings.extend(embeddings)
        print(f"  → Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks", end="\r")
    print(f"  → {len(all_embeddings)} embeddings generated              ")

    # Step 4: Store in ChromaDB
    print("\n[4/4] Storing in ChromaDB...")
    store = Store()
    store.reset()  # Clear existing index
    store.add_documents(chunks, all_embeddings)
    print(f"  → {store.count} chunks indexed")

    print("\n" + "=" * 60)
    print("  Index built successfully! 🎉")
    print("=" * 60)


def cmd_ask(args):
    """Ask a single question."""
    pipeline = Pipeline()
    result = pipeline.ask(args.question)

    print("\n" + "=" * 60)
    print(f"  Q: {result.query}")
    print("=" * 60)
    print(f"\n{result.answer}")

    if result.sources:
        print("\n📖 来源:")
        for src in result.sources:
            dist = src.get("distance")
            dist_str = f" (distance: {dist:.4f})" if dist is not None else ""
            print(f"  - {src['source']} 第{src['page']}页{dist_str}")


def cmd_interactive(args):
    """Interactive Q&A mode."""
    print("=" * 60)
    print("  Nidus RAG — Interactive Mode")
    print("  Type your questions (or 'quit' to exit)")
    print("=" * 60)

    pipeline = Pipeline()

    while True:
        try:
            query = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result = pipeline.ask(query)
        print(f"\n🤖 Nidus: {result.answer}")

        if result.sources:
            print("   📖 来源:", end=" ")
            source_strs = []
            for src in result.sources:
                source_strs.append(f"{src['source']} 第{src['page']}页")
            print(", ".join(source_strs))


def main():
    parser = argparse.ArgumentParser(
        prog="nidus",
        description="Nidus — Light Extensible Agentic RAG",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # index
    index_parser = subparsers.add_parser("index", help="Build document index")
    index_parser.set_defaults(func=cmd_index)

    # ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", type=str, help="Your question")
    ask_parser.set_defaults(func=cmd_ask)

    # interactive
    interactive_parser = subparsers.add_parser("interactive", help="Interactive Q&A mode")
    interactive_parser.set_defaults(func=cmd_interactive)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
