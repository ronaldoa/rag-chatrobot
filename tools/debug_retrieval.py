"""
Quick helper to inspect retrieval results for a given question.

Usage:
    python tools/debug_retrieval.py --question "What was the narrator's first method?"
    python tools/debug_retrieval.py -q "..." -k 5
"""
import argparse
from textwrap import shorten
from pathlib import Path
import sys

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qa_service import qa_service


def main():
    parser = argparse.ArgumentParser(description="Inspect top-K retrieved chunks for a question.")
    parser.add_argument("-q", "--question", required=True, help="Question to test retrieval on.")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Number of chunks to display.")
    args = parser.parse_args()

    qa_service.initialize()
    docs = qa_service._retriever.get_relevant_documents(args.question)[: args.top_k]  # type: ignore[attr-defined]

    if not docs:
        print("No documents returned.")
        return

    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        print(f"\n--- #{idx} ---")
        print(f"Source: {source} | Page: {page}")
        print(doc.page_content.strip())


if __name__ == "__main__":
    main()
