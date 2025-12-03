"""RAG evaluation using RAGAS only.

Usage:
    python -m src.eval_ragas --eval-path test.csv
    python -m src.eval_ragas --eval-path test.csv --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset
from ragas import evaluate as ragas_evaluate

# ragas metrics (answer_relevance was named answer_relevancy in some releases)
try:
    from ragas.metrics import answer_relevance, context_precision, context_recall, faithfulness
except ImportError:  # pragma: no cover - fallback for older ragas
    from ragas.metrics import (
        answer_relevancy as answer_relevance,  # type: ignore
        context_precision,
        context_recall,
        faithfulness,
    )

from .qa_service import qa_service


class RagasRAGEvaluator:
    """RAG evaluator that only uses RAGAS metrics."""

    def __init__(self) -> None:
        pass

    # ----------------- Data loading -----------------
    def load_eval_set(self, path: Path) -> List[Dict[str, Any]]:
        """Load evaluation samples from CSV or JSONL."""
        if path.suffix.lower() == ".csv":
            return self._load_csv(path)
        if path.suffix.lower() == ".jsonl":
            return self._load_jsonl(path)
        raise ValueError(f"Unsupported file format: {path.suffix}")

    def _load_csv(self, path: Path) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=1):
                question = (row.get("Question") or row.get("question") or "").strip()
                if not question:
                    continue
                expected = (row.get("Answer") or row.get("answer") or row.get("expected") or "").strip()
                label = (row.get("Label") or row.get("label") or row.get("category") or "").strip()
                samples.append(
                    {
                        "question": question,
                        "expected_answer": expected,
                        "label": label,
                        "row_idx": row_idx,
                    }
                )
        return samples

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                question = (data.get("question") or "").strip()
                expected = (data.get("expected_answer") or data.get("answer") or "").strip()
                label = (data.get("label") or "").strip()
                if not question:
                    continue
                samples.append(
                    {
                        "question": question,
                        "expected_answer": expected,
                        "label": label,
                        "row_idx": idx,
                    }
                )
        return samples

    # ----------------- Single sample eval -----------------
    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Call qa_service.ask(question) to get answer and sources."""
        question = sample.get("question", "").strip()
        expected = sample.get("expected_answer", "").strip()
        if not question:
            return {
                "error": "Empty question",
                "question": "",
                "expected_answer": expected,
                "model_answer": "",
                "sources": [],
                "row_idx": sample.get("row_idx", 0),
            }

        start = time.time()
        try:
            output = qa_service.ask(question)
            answer = output.get("answer", "")
            sources = output.get("sources", [])
            latency = time.time() - start

            return {
                "question": question,
                "expected_answer": expected,
                "model_answer": answer,
                "label": sample.get("label", ""),
                "row_idx": sample.get("row_idx", 0),
                "sources": sources,
                "latency_sec": round(latency, 3),
            }
        except Exception as e:  # noqa: BLE001
            latency = time.time() - start
            return {
                "question": question,
                "expected_answer": expected,
                "model_answer": f"ERROR: {str(e)[:200]}",
                "sources": [],
                "error": str(e),
                "latency_sec": round(latency, 3),
                "row_idx": sample.get("row_idx", 0),
            }

    # ----------------- Batch eval -----------------
    def evaluate_batch(self, samples: List[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        total = len(samples) if not limit else min(limit, len(samples))
        print(f"Evaluating {total} samples via qa_service.ask()...")

        for idx, sample in enumerate(samples):
            if limit and idx >= limit:
                break
            preview = sample.get("question", "")
            preview = preview[:50] + "..." if len(preview) > 50 else preview
            print(f"[{idx+1}/{total}] {preview}")
            result = self.evaluate_sample(sample)
            results.append(result)

            if "error" in result:
                print(f"    ✗ Error: {result.get('error', 'Unknown')}")
            else:
                print(
                    f"    ✓ Answer length: {len(result.get('model_answer', ''))} | "
                    f"Sources: {len(result.get('sources', []))} | "
                    f"Latency: {result.get('latency_sec', 0):.2f}s"
                )
        return results

    # ----------------- RAGAS evaluation -----------------
    def run_ragas(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Construct ragas dataset and run metrics."""
        valid = [r for r in results if "error" not in r]
        if not valid:
            print("No valid results for RAGAS evaluation.")
            return {}

        questions: List[str] = []
        answers: List[str] = []
        contexts: List[List[str]] = []
        ground_truths: List[str] = []

        for r in valid:
            q = r.get("question", "")
            a = r.get("model_answer", "")
            gt = r.get("expected_answer", "")

            srcs = r.get("sources", [])
            ctx_list: List[str] = []
            for s in srcs:
                if isinstance(s, dict):
                    text = s.get("text") or s.get("content") or s.get("page_content") or ""
                    if text:
                        ctx_list.append(str(text))
                elif s:
                    ctx_list.append(str(s))

            if not ctx_list:
                ctx_list = [""]

            questions.append(q)
            answers.append(a)
            contexts.append(ctx_list)
            ground_truths.append(gt)

        data_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }

        ds = Dataset.from_dict(data_dict)
        metrics = [answer_relevance, faithfulness, context_precision, context_recall]

        print("\nRunning RAGAS evaluation on valid samples...")
        ragas_result = ragas_evaluate(
            dataset=ds,
            metrics=metrics,
            llm=getattr(qa_service, "_llm", None),
            embeddings=getattr(qa_service, "_embeddings", None),
        )

        agg_scores = getattr(ragas_result, "scores", {}) if ragas_result else {}
        print("\nRAGAS METRICS")
        print("=" * 40)
        if isinstance(agg_scores, dict):
            for k, v in agg_scores.items():
                try:
                    print(f"{k:<20}: {float(v):.4f}")
                except Exception:
                    print(f"{k:<20}: {v}")
        else:
            try:
                for k, v in ragas_result.items():
                    print(f"{k:<20}: {float(v):.4f}")
            except Exception:
                print(agg_scores)
        print("=" * 40)
        print(f"Samples evaluated (RAGAS): {len(questions)}")

        try:
            df = ragas_result.to_pandas()
            df.to_csv("eval_results/ragas_results.csv", index=False)
            print("Per-sample ragas results saved to: eval_results/ragas_results.csv")
        except Exception:
            pass

        return agg_scores if isinstance(agg_scores, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Evaluation (RAGAS only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        required=True,
        help="Path to eval data (.csv or .jsonl, with Question/Answer/Label).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for evaluation.")

    args = parser.parse_args()

    evaluator = RagasRAGEvaluator()

    print(f"\nLoading data from: {args.eval_path}")
    samples = evaluator.load_eval_set(args.eval_path)
    print(f"Loaded {len(samples)} samples")
    if not samples:
        print("No samples found, exiting.")
        return

    qa_service.initialize()
    results = evaluator.evaluate_batch(samples, limit=args.limit)
    evaluator.run_ragas(results)

    print("\nEvaluation complete.\n")


if __name__ == "__main__":
    main()
