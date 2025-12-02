"""RAG evaluation runner with local GGUF LLM judge and YAML config.

Usage:
    python -m src.eval_local --eval-path eval.csv --config config/eval_config.yaml
    python -m src.eval_local --eval-path eval.csv --limit 10
    # CLI flags override config file values.
"""

import argparse
import csv
import json
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Optional dependencies
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = Any  # type: ignore
    util = None  # type: ignore

from .qa_service import qa_service


DEFAULT_CONFIG = {
    "evaluation": {
        "local_model": None,
        "embeddings_model": "all-MiniLM-L6-v2",
        "use_llm_judge": False,
        "limit": None,  # alias for max_samples
        "max_samples": None,
        "timeout_seconds": 30,
        "metrics": [
            "substring_match",
            "keyword_coverage",
            "semantic_similarity",
            "word_overlap",
        ],
        "output_dir": "eval_results",
        "save_results": True,
        "generate_report": True,
        "visualize": False,
    }
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dict."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base


def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load YAML config and merge with defaults."""
    cfg = DEFAULT_CONFIG.copy()
    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            cfg = _deep_update(cfg, data)
    return cfg


class LocalLlamaJudge:
    """LLM judge using local GGUF model via llama.cpp."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is required for local model inference")

        print(f"Loading local model: {model_path}")
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            n_threads=4,
            n_batch=512,
        )
        print("Model loaded successfully")

    def judge_answer(self, question: str, expected: str, answer: str, context: str | None = None) -> Dict[str, Any]:
        prompt = self._create_prompt(question, expected, answer, context)
        try:
            start = time.time()
            response = self.model(
                prompt,
                max_tokens=256,
                temperature=0.1,
                stop=["</evaluation>", "\n\n"],
                echo=False,
            )
            latency = time.time() - start
            text = response.get("choices", [{}])[0].get("text", "") if isinstance(response, dict) else str(response)
            scores, reason = self._parse_response(text)
            return {
                "llm_scores": scores,
                "llm_reason": reason,
                "judge_latency": round(latency, 3),
            }
        except Exception as e:  # noqa: BLE001
            return {
                "llm_scores": {"overall": 0.0},
                "llm_reason": f"Judge error: {str(e)[:100]}",
                "judge_latency": 0.0,
            }

    def _create_prompt(self, question: str, expected: str, answer: str, context: str | None) -> str:
        return f"""<|start_header_id|>system<|end_header_id|>

You are a professional QA system evaluator. Score each criterion from 0.0 to 1.0.

Output format:
ACCURACY: <score>
COMPLETENESS: <score>
FAITHFULNESS: <score>
CLARITY: <score>
OVERALL: <average_score>
REASON: <brief_explanation>

<|eot_id|><|start_header_id|>user<|end_header_id|>

QUESTION: {question}

REFERENCE ANSWER: {expected}

MODEL ANSWER: {answer}

{f"CONTEXT PROVIDED: {context}" if context else "CONTEXT: Not provided"}

Please evaluate the MODEL ANSWER based on the REFERENCE ANSWER and CONTEXT (if provided).<|eot_id|><|start_header_id|>assistant<|end_header_id|>

EVALUATION:
"""

    def _parse_response(self, text: str) -> Tuple[Dict[str, float], str]:
        scores: Dict[str, float] = {}
        patterns = {
            "accuracy": r"ACCURACY\s*[:=]\s*([0-9]*\.?[0-9]+)",
            "completeness": r"COMPLETENESS\s*[:=]\s*([0-9]*\.?[0-9]+)",
            "faithfulness": r"FAITHFULNESS\s*[:=]\s*([0-9]*\.?[0-9]+)",
            "clarity": r"CLARITY\s*[:=]\s*([0-9]*\.?[0-9]+)",
            "overall": r"OVERALL\s*[:=]\s*([0-9]*\.?[0-9]+)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    scores[key] = float(match.group(1))
                except ValueError:
                    scores[key] = 0.0

        reason_match = re.search(r"REASON\s*[:=]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else ""

        if "overall" not in scores and scores:
            scores["overall"] = round(sum(scores.values()) / len(scores), 3)

        for k in scores:
            scores[k] = max(0.0, min(1.0, scores[k]))

        return scores, reason[:200]


class LightweightRAGEvaluator:
    """Lightweight evaluator that can use a local LLM judge."""

    def __init__(
        self,
        local_model_path: str | None = None,
        use_llm_judge: bool = False,
        embeddings_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.use_llm_judge = use_llm_judge
        self.embeddings_model_name = embeddings_model
        self.embeddings_model: SentenceTransformer | None = None
        self.llm_judge: LocalLlamaJudge | None = None

        if use_llm_judge and local_model_path:
            if LLAMA_CPP_AVAILABLE:
                try:
                    self.llm_judge = LocalLlamaJudge(local_model_path)
                except Exception as e:  # noqa: BLE001
                    print(f"Failed to load local model: {e}")
            else:
                print("llama-cpp-python not available, LLM judge disabled")

    # ----------------- Data loading -----------------
    def load_eval_set(self, path: Path) -> List[Dict[str, Any]]:
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
                samples.append(json.loads(line))
        return samples

    # ----------------- Basic metrics -----------------
    def evaluate_without_llm(self, question: str, expected: str, answer: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics["substring_match"] = bool(expected) and (expected.lower() in answer.lower())
        metrics["keyword_coverage"] = self._keyword_coverage(expected, answer)
        metrics["word_overlap"] = self._word_overlap(expected, answer)

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            sim = self._semantic_similarity(expected, answer)
            metrics["semantic_similarity"] = sim

        if expected:
            metrics["length_ratio"] = len(answer) / max(len(expected), 1)

        return metrics

    def _keyword_coverage(self, expected: str, answer: str) -> float:
        if not expected or not answer:
            return 0.0
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        expected_words = set(re.findall(r"\b\w+\b", expected.lower()))
        expected_words = {w for w in expected_words if w not in stopwords and len(w) > 3}
        if not expected_words:
            return 0.0
        answer_lower = answer.lower()
        covered = sum(1 for w in expected_words if w in answer_lower)
        return covered / len(expected_words)

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        model = self._get_embeddings_model()
        if model is None:
            return 0.0
        try:
            emb = model.encode([text1, text2], convert_to_tensor=True, normalize_embeddings=True)
            return float(util.cos_sim(emb[0], emb[1]))
        except Exception as e:  # noqa: BLE001
            print(f"Embeddings error: {e}")
            return 0.0

    def _word_overlap(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        w1 = set(re.findall(r"\b\w+\b", text1.lower()))
        w2 = set(re.findall(r"\b\w+\b", text2.lower()))
        union = w1 | w2
        if not union:
            return 0.0
        return len(w1 & w2) / len(union)

    def _get_embeddings_model(self) -> SentenceTransformer | None:
        if self.embeddings_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading embeddings model: {self.embeddings_model_name}")
                self.embeddings_model = SentenceTransformer(self.embeddings_model_name)
            except Exception as e:  # noqa: BLE001
                print(f"Failed to load embeddings model: {e}")
                self.embeddings_model = None
        return self.embeddings_model

    # ----------------- Sample evaluation -----------------
    def evaluate_sample(self, sample: Dict[str, Any], qa_service_func=None) -> Dict[str, Any]:
        question = sample.get("question", "").strip()
        expected = sample.get("expected_answer", "").strip()
        if not question:
            return {"error": "Empty question"}

        start = time.time()
        try:
            if qa_service_func:
                output = qa_service_func(question)
            else:
                qa_service.initialize()
                output = qa_service.ask(question)

            answer = output.get("answer", "")
            sources = output.get("sources", [])
            latency = time.time() - start

            metrics = self.evaluate_without_llm(question, expected, answer)
            retrieval_metrics = self._evaluate_retrieval(sources, question)

            result: Dict[str, Any] = {
                "question": question,
                "expected_answer": expected,
                "model_answer": answer,
                "label": sample.get("label", ""),
                "row_idx": sample.get("row_idx", 0),
                **metrics,
                **retrieval_metrics,
                "latency_sec": round(latency, 3),
                "num_sources": len(sources),
                "answer_length": len(answer),
                "expected_length": len(expected),
            }

            if self.use_llm_judge and self.llm_judge and expected:
                context_text = "\n".join(
                    (s.get("text") or s.get("content") or "") if isinstance(s, dict) else str(s)
                    for s in sources
                )
                judge = self.llm_judge.judge_answer(question, expected, answer, context_text[:1000])
                result.update(
                    {
                        "llm_overall": judge.get("llm_scores", {}).get("overall", 0.0),
                        "llm_accuracy": judge.get("llm_scores", {}).get("accuracy", 0.0),
                        "llm_completeness": judge.get("llm_scores", {}).get("completeness", 0.0),
                        "llm_faithfulness": judge.get("llm_scores", {}).get("faithfulness", 0.0),
                        "llm_clarity": judge.get("llm_scores", {}).get("clarity", 0.0),
                        "llm_reason": judge.get("llm_reason", ""),
                        "judge_latency": judge.get("judge_latency", 0.0),
                    }
                )

            return result

        except Exception as e:  # noqa: BLE001
            latency = time.time() - start
            return {
                "question": question,
                "expected_answer": expected,
                "model_answer": f"ERROR: {str(e)[:200]}",
                "error": str(e),
                "latency_sec": round(latency, 3),
                "row_idx": sample.get("row_idx", 0),
            }

    def _evaluate_retrieval(self, sources: List[Any], question: str) -> Dict[str, float]:
        if not sources:
            return {"avg_relevance": 0.0, "max_relevance": 0.0}
        q_keywords = {w for w in re.findall(r"\b\w+\b", question.lower()) if len(w) > 3}
        scores: List[float] = []
        for src in sources:
            text = src.get("text") or src.get("content") or "" if isinstance(src, dict) else str(src)
            if not text or not q_keywords:
                scores.append(0.0)
                continue
            text_lower = text.lower()
            matches = sum(1 for kw in q_keywords if kw in text_lower)
            scores.append(matches / len(q_keywords))
        return {
            "avg_relevance": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "max_relevance": round(max(scores), 3) if scores else 0.0,
        }

    # ----------------- Batch evaluation -----------------
    def evaluate_batch(self, samples: List[Dict[str, Any]], limit: Optional[int] = None, qa_service_func=None) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        total = len(samples) if not limit else min(limit, len(samples))
        print(f"Evaluating {total} samples...")
        for idx, sample in enumerate(samples):
            if limit and idx >= limit:
                break
            preview = sample.get("question", "")
            preview = preview[:50] + "..." if len(preview) > 50 else preview
            print(f"[{idx+1}/{total}] {preview}")
            result = self.evaluate_sample(sample, qa_service_func)
            results.append(result)
            if "error" not in result:
                match = "✓" if result.get("substring_match") else "✗"
                sim = result.get("semantic_similarity", 0.0)
                print(f"     {match} Sim: {sim:.3f} | Latency: {result.get('latency_sec', 0):.2f}s")
            else:
                print(f"     ✗ Error: {result.get('error', 'Unknown')}")
        return results

    # ----------------- Summaries & IO -----------------
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"total_samples": 0}

        valid = [r for r in results if "error" not in r]
        errors = [r for r in results if "error" in r]
        summary: Dict[str, Any] = {
            "total_samples": len(results),
            "valid_samples": len(valid),
            "error_samples": len(errors),
            "error_rate": len(errors) / len(results) if results else 0.0,
        }

        if not valid:
            return summary

        def _avg(metric: str) -> float:
            vals = [r.get(metric) for r in valid if r.get(metric) is not None]
            return round(statistics.mean(vals), 3) if vals else 0.0

        summary["substring_match_rate"] = round(
            sum(1 for r in valid if r.get("substring_match")) / len(valid), 3
        )
        for metric in ["keyword_coverage", "semantic_similarity", "word_overlap", "avg_relevance", "max_relevance"]:
            summary[f"avg_{metric}"] = _avg(metric)

        latencies = [r.get("latency_sec", 0) for r in valid]
        summary["avg_latency"] = round(statistics.mean(latencies), 3) if latencies else 0.0

        llm_scores = [r.get("llm_overall") for r in valid if r.get("llm_overall") is not None]
        if llm_scores:
            summary["avg_llm_score"] = round(statistics.mean(llm_scores), 3)
            summary["llm_score_std"] = round(statistics.stdev(llm_scores), 3) if len(llm_scores) > 1 else 0.0

        label_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "scores": []})
        for r in valid:
            label = r.get("label", "Unknown")
            label_stats[label]["count"] += 1
            score = r.get("semantic_similarity")
            if score is not None:
                label_stats[label]["scores"].append(score)

        for label, stats in label_stats.items():
            if stats["scores"]:
                stats["avg_score"] = round(statistics.mean(stats["scores"]), 3)
                if len(stats["scores"]) > 1:
                    stats["std_score"] = round(statistics.stdev(stats["scores"]) ,3)

        summary["by_label"] = dict(label_stats)
        return summary

    def save_results_csv(self, results: List[Dict[str, Any]], output_path: Path) -> None:
        if not results:
            print("No results to save")
            return
        fieldnames: set[str] = set()
        for r in results:
            fieldnames.update(r.keys())
        preferred = [
            "row_idx",
            "label",
            "question",
            "expected_answer",
            "model_answer",
            "substring_match",
            "keyword_coverage",
            "semantic_similarity",
            "word_overlap",
            "length_ratio",
            "llm_overall",
            "llm_accuracy",
            "llm_completeness",
            "llm_faithfulness",
            "llm_clarity",
            "llm_reason",
            "avg_relevance",
            "max_relevance",
            "num_sources",
            "latency_sec",
            "judge_latency",
            "answer_length",
            "expected_length",
            "error",
        ]
        ordered = [f for f in preferred if f in fieldnames] + sorted(fieldnames - set(preferred))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ordered)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Results saved to: {output_path}")

    def generate_report(self, summary: Dict[str, Any], output_path: Path) -> None:
        lines = [
            "=" * 60,
            "RAG EVALUATION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "OVERALL",
            "-" * 40,
            f"Total samples: {summary.get('total_samples', 0)}",
            f"Valid samples: {summary.get('valid_samples', 0)}",
            f"Error samples: {summary.get('error_samples', 0)}",
            f"Error rate: {summary.get('error_rate', 0):.1%}",
            "",
            "METRICS",
            "-" * 40,
            f"Substring Match: {summary.get('substring_match_rate', 0):.1%}",
            f"Keyword Coverage: {summary.get('avg_keyword_coverage', 0):.1%}",
            f"Semantic Similarity: {summary.get('avg_semantic_similarity', 0):.3f}",
            f"Word Overlap: {summary.get('avg_word_overlap', 0):.3f}",
            f"Avg Latency: {summary.get('avg_latency', 0):.3f}s",
        ]
        if summary.get("avg_llm_score") is not None:
            lines.append(f"LLM Judge Score: {summary['avg_llm_score']:.3f}")
        if summary.get("by_label"):
            lines.extend(["", "BY LABEL", "-" * 40])
            for label, stats in summary["by_label"].items():
                avg_score = stats.get("avg_score", 0)
                lines.append(f"{label}: {stats.get('count', 0)} samples, avg semantic: {avg_score:.3f}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Report saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Evaluation for Local GGUF Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--eval-path", type=Path, required=True, help="Path to eval data (.csv or .jsonl)")
    parser.add_argument("--config", type=Path, default=Path("config/eval_config.yaml"), help="YAML config path")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path (overrides config output_dir)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--llm-judge", action="store_true", help="Enable LLM judge")
    parser.add_argument("--local-model", type=Path, help="Path to local GGUF model for LLM judge")
    parser.add_argument("--embeddings", type=str, default=None, help="Sentence transformers model")
    parser.add_argument("--report", action="store_true", help="Generate summary report")
    parser.add_argument("--report-path", type=Path, default=None, help="Report output path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_eval = cfg.get("evaluation", {})

    use_llm_judge = args.llm_judge or cfg_eval.get("use_llm_judge", False)
    local_model = str(args.local_model) if args.local_model else cfg_eval.get("local_model")
    embeddings_model = args.embeddings or cfg_eval.get("embeddings_model", "all-MiniLM-L6-v2")
    limit = args.limit
    if limit is None:
        limit = cfg_eval.get("limit")
    if limit is None:
        limit = cfg_eval.get("max_samples")
    output_dir = Path(cfg_eval.get("output_dir", "eval_results"))
    output_csv = args.out or (output_dir / "results.csv")
    report_path = args.report_path or (output_dir / "report.txt")
    generate_report = args.report or cfg_eval.get("generate_report", True)

    if use_llm_judge and not local_model:
        raise ValueError("--llm-judge enabled but no local model provided in args or config")

    evaluator = LightweightRAGEvaluator(
        local_model_path=local_model,
        use_llm_judge=use_llm_judge,
        embeddings_model=embeddings_model,
    )

    print(f"\nLoading data from: {args.eval_path}")
    samples = evaluator.load_eval_set(args.eval_path)
    print(f"Loaded {len(samples)} samples")
    if not samples:
        print("No samples found, exiting")
        return

    def qa_func(q: str) -> Dict[str, Any]:
        qa_service.initialize()
        return qa_service.ask(q)

    results = evaluator.evaluate_batch(samples, limit=limit, qa_service_func=qa_func)
    summary = evaluator.generate_summary(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.save_results_csv(results, output_csv)
    if generate_report:
        evaluator.generate_report(summary, report_path)

    print("\nQUICK SUMMARY")
    print("=" * 40)
    print(f"Samples: {summary.get('total_samples', 0)} (valid {summary.get('valid_samples', 0)})")
    print(f"Substring Match: {summary.get('substring_match_rate', 0):.1%}")
    if "avg_semantic_similarity" in summary:
        print(f"Semantic Similarity: {summary['avg_semantic_similarity']:.3f}")
    if summary.get("avg_llm_score") is not None:
        print(f"LLM Judge Score: {summary['avg_llm_score']:.3f}")
    print(f"Avg Latency: {summary.get('avg_latency', 0):.3f}s")
    print("Evaluation complete!\n")


if __name__ == "__main__":
    main()
