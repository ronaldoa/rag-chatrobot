from src.qa_service import qa_service


def debug_one(question: str):
    qa_service.initialize()

    out = qa_service.ask(question)

    print("=== QUESTION ===")
    print(question)
    print("\n=== MODEL ANSWER ===")
    print(out["answer"])

    print("\n=== SOURCES ===")
    for i, doc in enumerate(out["sources"], 1):
        src = doc.get("source")
        page = doc.get("page")
        content = doc.get("content") or doc.get("page_content") or ""
        print(f"\n--- Source {i} ({src}, page {page}) ---")
        print(content[:800])


if __name__ == "__main__":
    q = "How did Livingston’s winning bucket shop method fail when he used it in a real brokerage office?"
    q1 = "Why did he need the tape to be absolutely current?"
    q2 = "When did the narrator first go to work as a quotation-board boy in a stock-brokerage office?"
    q3 = "when did i go to work?"
    q4 = "In the episode where the narrator explains that in a brokerage office the tape was 'ancient history,' why did he say he needed the tape to be absolutely current, and how did even a small delay in the quoted price make his timing method unreliable?"

    debug_one(q3)
