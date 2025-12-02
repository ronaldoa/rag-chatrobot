"""
Lightweight RAG evaluation runner - supports CSV format

Usage:
    python -m src.eval --eval-path eval.csv --out results.csv --limit 100

Input file (CSV):
    Question,Answer,Label
    "What was...", "He noticed that...", "Timing"
"""
import argparse
import csv
import re
import time
from pathlib import Path
from typing import List, Dict, Any

from .qa_service import qa_service
from .llm import get_llm
from sentence_transformers import SentenceTransformer, util


def load_eval_set_csv(path: Path) -> List[Dict[str, Any]]:
    """
    Load evaluation samples from a CSV file.

    CSV format:
        Question,Answer,Label
        "Question 1", "Answer 1", "Category 1"
        "Question 2", "Answer 2", "Category 2"
    """
    samples: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader, start=1):
            # Retrieve fields (fallback to alternate column names)
            question = (
                row.get("Question") or
                row.get("question") or
                ""
            ).strip()

            answer = (
                row.get("Answer") or
                row.get("answer") or
                row.get("expected") or
                ""
            ).strip()

            label = (
                row.get("Label") or
                row.get("label") or
                row.get("category") or
                ""
            ).strip()

            if not question:
                print(f"Warning: skip row {row_idx}: question is empty")
                continue

            samples.append({
                "question": question,
                "answer": answer,      # expected answer
                "label": label,        # preserve label information
                "row_idx": row_idx
            })

    return samples


def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    """Auto-detect file format and load samples."""
    if path.suffix.lower() == '.csv':
        return load_eval_set_csv(path)
    elif path.suffix.lower() == '.jsonl':
        # Original JSONL loading logic
        import json
        samples: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON: {e}")
        return samples
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}; please use .csv or .jsonl")


def _run_llm_judge(llm, question: str, expected: str, answer: str) -> Dict[str, Any]:
    """
    Use the LLM to score answer correctness (0-1).

    Returns: {"score": float, "reason": str}
    """
    prompt = (
        "You are a strict grader for QA. "
        "Given a question, a reference answer, and a model answer, score correctness from 0 to 1. "
        "Output exactly: 'score: <0-1> reason: <short reason>'.\n\n"
        f"Question: {question}\n"
        f"Reference answer: {expected}\n"
        f"Model answer: {answer}\n"
    )
    try:
        resp = llm(prompt)
        text = resp if isinstance(resp, str) else resp.get("choices", [{}])[0].get("text", "")
    except Exception as e:
        return {"score": 0.0, "reason": f"judge_error: {e}"}

    matches = re.findall(r"score\s*[:=]\s*(-?\d+(?:\.\d+)?)", str(text), flags=re.IGNORECASE)
    score = float(matches[-1]) if matches else 0.0
    return {
        "score": max(0.0, min(1.0, score)),
        "reason": str(text).strip()[:500]
    }


_semantic_model: SentenceTransformer | None = None


def _get_semantic_model() -> SentenceTransformer:
    """Lazy-load sentence embedding model for semantic similarity."""
    global _semantic_model
    if _semantic_model is None:
        # Reuse the same model as embeddings to avoid extra download
        _semantic_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    return _semantic_model


def semantic_similarity(expected: str, answer: str) -> float:
    """Compute cosine similarity between expected and answer embeddings."""
    if not expected or not answer:
        return 0.0
    model = _get_semantic_model()
    emb = model.encode([expected, answer], convert_to_tensor=True, normalize_embeddings=True)
    score = float(util.cos_sim(emb[0], emb[1]))
    return max(0.0, min(1.0, score))


def rouge_l(expected: str, answer: str) -> float:
    """Compute a simple ROUGE-L (LCS-based F1)."""
    if not expected or not answer:
        return 0.0

    def lcs(a: list[str], b: list[str]) -> int:
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a)):
            for j in range(len(b)):
                if a[i] == b[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
        return dp[-1][-1]

    ref_tokens = expected.split()
    hyp_tokens = answer.split()
    lcs_len = lcs(ref_tokens, hyp_tokens)
    if lcs_len == 0:
        return 0.0
    recall = lcs_len / len(ref_tokens)
    precision = lcs_len / len(hyp_tokens)
    f1 = 2 * recall * precision / (recall + precision)
    return f1


def evaluate(samples: List[Dict[str, Any]], limit: int | None = None, use_llm_judge: bool = False) -> List[Dict[str, Any]]:
    """Run evaluation."""
    qa_service.initialize()
    judge_llm = None
    if use_llm_judge:
        # Reuse initialized LLM if present; otherwise load a lightweight variant
        judge_llm = getattr(qa_service, "_llm", None) or get_llm(overrides={"max_tokens": 64, "temperature": 0.1})
    results: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        if limit and idx >= limit:
            break

        question = sample.get("question", "").strip()
        expected = sample.get("answer", "").strip()
        label = sample.get("label", "")

        if not question:
            continue

        print(f"[{idx+1}/{len(samples) if not limit else min(limit, len(samples))}] {question[:60]}...")

        start = time.time()
        try:
            output = qa_service.ask(question)
            answer = output.get("answer", "")
            sources = output.get("sources", [])

            # Evaluation metrics
            # 1. Simple substring match
            substring_match = bool(expected) and (expected.lower() in answer.lower())

            # 2. Keyword coverage
            keyword_coverage = calculate_keyword_coverage(expected, answer)

            # 3. Semantic similarity + ROUGE-L
            sem_sim = semantic_similarity(expected, answer)
            rouge_l_score = rouge_l(expected, answer)

            # 3. LLM judge
            judge_score = None
            judge_reason = None
            if use_llm_judge and judge_llm:
                judge = _run_llm_judge(judge_llm, question, expected, answer)
                judge_score = judge.get("score")
                judge_reason = judge.get("reason")

            latency = time.time() - start

            results.append({
                "idx": idx,
                "row_idx": sample.get("row_idx", idx),
                "label": label,
                "question": question,
                "expected": expected,
                "answer": answer,
                "substring_match": substring_match,
                "keyword_coverage": keyword_coverage,
                "semantic_similarity": sem_sim,
                "rouge_l": rouge_l_score,
                "latency_sec": round(latency, 3),
                "num_sources": len(sources),
                "sources": [s.get("source", "") for s in sources],
                "judge_score": judge_score,
                "judge_reason": judge_reason,
            })

            print(f"  Match: {substring_match} | Keyword: {keyword_coverage:.2f} | Latency: {latency:.2f}s")

        except Exception as e:
            latency = time.time() - start
            results.append({
                "idx": idx,
                "row_idx": sample.get("row_idx", idx),
                "label": label,
                "question": question,
                "expected": expected,
                "answer": f"ERROR: {e}",
                "substring_match": False,
                "keyword_coverage": 0.0,
                "semantic_similarity": 0.0,
                "rouge_l": 0.0,
                "latency_sec": round(latency, 3),
                "num_sources": 0,
                "sources": [],
                "judge_score": None,
                "judge_reason": f"error: {e}",
            })
            print(f"  Error: {e}")

    return results


def calculate_keyword_coverage(expected: str, answer: str) -> float:
    """
    Calculate coverage of expected keywords within the generated answer.

    Returns:
        Score between 0.0 and 1.0
    """
    if not expected or not answer:
        return 0.0

    # Simple tokenization (replace with a richer tokenizer if needed)
    import re

    # Extract keywords from the expected answer (drop common stopwords)
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'he', 'she', 'it', 'they', 'his', 'her', 'their', 'this', 'that',
        'have'
    }

    expected_words = set(re.findall(r'\w+', expected.lower()))
    expected_words = {w for w in expected_words if w not in stopwords and len(w) > 2}

    if not expected_words:
        return 0.0

    answer_lower = answer.lower()

    # Count covered keywords
    covered = sum(1 for word in expected_words if word in answer_lower)

    return covered / len(expected_words)


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Write evaluation results to CSV."""
    fieldnames = [
        "idx", "row_idx", "label", "question", "expected", "answer",
        "substring_match", "keyword_coverage", "semantic_similarity", "rouge_l",
        "judge_score", "judge_reason", "latency_sec", "num_sources", "sources"
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Flatten list of sources to a string
            if isinstance(row.get("sources"), list):
                row["sources"] = "; ".join(row["sources"])
            writer.writerow(row)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics."""
    total = len(rows)
    if total == 0:
        return {}

    # Basic statistics
    substring_matches = sum(1 for r in rows if r.get("substring_match"))
    avg_keyword_coverage = sum(r.get("keyword_coverage", 0.0) for r in rows) / total
    avg_semantic_similarity = sum(r.get("semantic_similarity", 0.0) for r in rows) / total
    avg_rouge_l = sum(r.get("rouge_l", 0.0) for r in rows) / total
    avg_latency = sum(r.get("latency_sec", 0.0) for r in rows) / total

    judge_scores = [r.get("judge_score") for r in rows if isinstance(r.get("judge_score"), (int, float))]
    judge_avg = round(sum(judge_scores) / len(judge_scores), 3) if judge_scores else None

    # Aggregate by label
    label_stats = {}
    for row in rows:
        label = row.get("label", "Unknown")
        if label not in label_stats:
            label_stats[label] = {
                "count": 0,
                "substring_matches": 0,
                "avg_coverage": 0.0,
                "avg_semantic": 0.0,
                "avg_rouge_l": 0.0,
                "avg_latency": 0.0
            }

        label_stats[label]["count"] += 1
        if row.get("substring_match"):
            label_stats[label]["substring_matches"] += 1
        label_stats[label]["avg_coverage"] += row.get("keyword_coverage", 0.0)
        label_stats[label]["avg_semantic"] += row.get("semantic_similarity", 0.0)
        label_stats[label]["avg_rouge_l"] += row.get("rouge_l", 0.0)
        label_stats[label]["avg_latency"] += row.get("latency_sec", 0.0)

    # Compute averages
    for label, stats in label_stats.items():
        count = stats["count"]
        stats["match_rate"] = round(stats["substring_matches"] / count, 3)
        stats["avg_coverage"] = round(stats["avg_coverage"] / count, 3)
        stats["avg_semantic"] = round(stats["avg_semantic"] / count, 3)
        stats["avg_rouge_l"] = round(stats["avg_rouge_l"] / count, 3)
        stats["avg_latency"] = round(stats["avg_latency"] / count, 3)

    return {
        "total": total,
        "substring_matches": substring_matches,
        "substring_match_rate": round(substring_matches / total, 3),
        "avg_keyword_coverage": round(avg_keyword_coverage, 3),
        "avg_semantic_similarity": round(avg_semantic_similarity, 3),
        "avg_rouge_l": round(avg_rouge_l, 3),
        "avg_latency_sec": round(avg_latency, 3),
        "by_label": label_stats,
        "judge_avg": judge_avg
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a RAG system (supports CSV and JSONL)")
    parser.add_argument("--eval-path", type=Path, required=True,
                       help="Path to evaluation data (.csv or .jsonl)")
    parser.add_argument("--out", type=Path, default=Path("eval_results.csv"),
                       help="Path to save results")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit the number of samples to evaluate")
    parser.add_argument("--llm-judge", action="store_true",
                       help="Enable LLM judging (returns 0-1 score and reason)")
    args = parser.parse_args()

    print(f"\nLoading data: {args.eval_path}")
    samples = load_eval_set(args.eval_path)
    print(f"Loaded {len(samples)} questions\n")

    print("Starting evaluation...\n")
    rows = evaluate(samples, limit=args.limit, use_llm_judge=args.llm_judge)

    print("\nGenerating stats...")
    stats = summarize(rows)

    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print(f"Total samples:   {stats['total']}")
    print(f"Substring match: {stats['substring_match_rate']:.1%}")
    print(f"Keyword cover:   {stats['avg_keyword_coverage']:.1%}")
    print(f"Semantic sim:    {stats['avg_semantic_similarity']:.3f}")
    print(f"ROUGE-L:         {stats['avg_rouge_l']:.3f}")
    print(f"Avg latency:     {stats['avg_latency_sec']:.3f}s")
    if stats.get("judge_avg") is not None:
        print(f"LLM score avg:   {stats['judge_avg']:.3f}")

    if stats.get('by_label'):
        print("\nBy label:")
        for label, label_stats in stats['by_label'].items():
            print(f"\n  [{label}]")
            print(f"    Count:     {label_stats['count']}")
            print(f"    Match:     {label_stats['match_rate']:.1%}")
            print(f"    Coverage:  {label_stats['avg_coverage']:.1%}")
            print(f"    Semantic:  {label_stats['avg_semantic']:.3f}")
            print(f"    ROUGE-L:   {label_stats['avg_rouge_l']:.3f}")
            print(f"    Avg delay: {label_stats['avg_latency']:.3f}s")

    print("="*60 + "\n")

    write_csv(rows, args.out)
    print(f"Results saved to: {args.out}\n")


if __name__ == "__main__":
    main()
