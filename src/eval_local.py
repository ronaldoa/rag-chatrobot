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

# Optional ragas evaluation
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    ragas_evaluate = None  # type: ignore
    answer_relevancy = context_precision = context_recall = faithfulness = None  # type: ignore

from .qa_service import qa_service


DEFAULT_CONFIG = {
    "evaluation": {
        "local_model": None,
        "embeddings_model": "bge-base-en-v1.5",
        "use_llm_judge": False,
        "use_ragas": False,
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

    def judge_answer(
        self,
        question: str,
        expected: str,
        answer: str,
        context: str | None = None,
    ) -> Dict[str, Any]:
        if context:
            # 如果你有 truncate 函数可以用，没有的话可以简单切一刀
            context = context[:12000]

        prompt = self._create_prompt(question, expected, answer, context)

        try:
            start = time.time()
            response = self.model(
                prompt,
                max_tokens=512,
                temperature=0.1,
                stop=["<|eot_id|>"],
                echo=False,
            )
            latency = time.time() - start

            text = response.get("choices", [{}])[0].get("text", "") if isinstance(response, dict) else str(response)

            # 调试建议：先看看模型到底输出了什么
            # print("DEBUG raw judge output:", repr(text[:300]))

            scores, reason = self._parse_response(text)

            return {
                "llm_scores": scores,
                "llm_reason": reason,
                "judge_latency": round(latency, 3),
            }
        except Exception as e:  # noqa: BLE001
            return {
                "llm_scores": {
                    "overall": 0.0,
                    "accuracy": 0.0,
                    "completeness": 0.0,
                    "faithfulness": 0.0,
                    "clarity": 0.0,
                },
                "llm_reason": f"Judge error: {str(e)[:100]}",
                "judge_latency": 0.0,
            }





    def _create_prompt(self, question: str, expected: str, answer: str, context: str | None) -> str:
        context_block = context if context else "[No context was provided to the model]"
        return f"""<|start_header_id|>system<|end_header_id|>

    You are a professional QA system evaluator.
    You MUST respond with ONLY a valid JSON object containing exactly these keys:
    "accuracy", "completeness", "faithfulness", "clarity", "reason"

    All scores must be floats between 0.0 and 1.0.

    Scoring Guidelines:

    1. ACCURACY (0.0-1.0): Does MODEL ANSWER match the facts in REFERENCE ANSWER?
    - 1.0: All key facts correct
    - 0.5: Partially correct, some errors
    - 0.0: Wrong or missing key facts

    2. COMPLETENESS (0.0-1.0): Does answer address all parts of the question?
    - 1.0: Fully answers the question
    - 0.5: Partially answers, missing details
    - 0.0: Does not answer the question

    3. FAITHFULNESS (0.0-1.0): Is EVERY claim in MODEL ANSWER supported by CONTEXT PROVIDED?
    - 1.0: All claims directly stated or clearly implied in context
    - 0.7: Most claims supported, minor acceptable inferences
    - 0.3: Some claims lack context support
    - 0.0: Major claims contradict or fabricate information not in context

    NOTE: If CONTEXT PROVIDED is "[No context was provided to the model]",
    focus on ACCURACY and COMPLETENESS vs REFERENCE ANSWER.
    In this case, you may set FAITHFULNESS to a neutral value (e.g., around 0.5)
    if the answer is plausible and consistent with the REFERENCE ANSWER.

    ⚠️ CRITICAL RULES:
    - If MODEL ANSWER says "did not say/mention/state" or directly denies something:
      Check if the CONTEXT actually contains contradicting information.
      If not, score FAITHFULNESS ≤ 0.3
    - Fabricated quotes or specific facts not in CONTEXT → score 0.0-0.3
    - Reasonable paraphrasing of CONTEXT → score 0.9-1.0

    4. CLARITY (0.0-1.0): Is the answer well-organized and easy to understand?
    - 1.0: Very clear and well-structured
    - 0.5: Somewhat clear but could be better
    - 0.0: Confusing or incoherent


    Example Output:
    {{
        "accuracy": 0.9,
        "completeness": 0.8,
        "faithfulness": 0.9,
        "clarity": 1.0,
        "reason": "Answer is accurate and complete, and all claims are supported by the context."
    }}

    <|eot_id|><|start_header_id|>user<|end_header_id|>

    QUESTION:
    {question}

    REFERENCE ANSWER:
    {expected}

    MODEL ANSWER:
    {answer}

    CONTEXT PROVIDED:
    {context_block}

    Evaluate the MODEL ANSWER. Output ONLY the JSON object, no other text.
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """


    def _parse_response(self, text: str) -> Tuple[Dict[str, float], str]:
        # 默认值
        scores = {
            "accuracy": 0.0,
            "completeness": 0.0,
            "faithfulness": 0.0,
            "clarity": 0.0,
            "overall": 0.0,
        }
        reason = "Parse error: no JSON parsed"

        try:
            # 1) 尝试从输出中抓出 JSON 块
            json_match = re.search(r"\{.*?\}", text, re.DOTALL)
            candidate = None

            if json_match:
                candidate = json_match.group(0).strip()
            else:
                # 2) 兜底：有些模型可能只输出了 JSON body，没有外层 {}
                stripped = text.strip()
                if stripped.startswith('"accuracy"') or stripped.startswith("'accuracy'"):
                    candidate = "{" + stripped.rstrip(", \n") + "}"

            if not candidate:
                reason = f"No JSON found. Raw: {text[:200]!r}"
                return scores, reason

            data = json.loads(candidate)

            for key in ["accuracy", "completeness", "faithfulness", "clarity"]:
                if key in data:
                    try:
                        scores[key] = float(data[key])
                    except Exception:
                        pass

            vals = [scores[k] for k in ["accuracy", "completeness", "faithfulness", "clarity"]]
            scores["overall"] = round(sum(vals) / len(vals), 3)
            reason = data.get("reason", f"Parsed successfully. Raw: {candidate[:120]!r}")
            return scores, reason

        except Exception as e:
            reason = f"JSON Parse Error: {str(e)} | Raw: {text[:200]!r}"
            return scores, reason



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
                judge = self.llm_judge.judge_answer(question, expected, answer, context_text)
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


def generate_plots(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """Generate simple PNG charts from evaluation results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping visualization")
        return

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No valid results to visualize")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    def _plot_hist(metric: str, title: str, bins: int = 20) -> None:
        vals = [r.get(metric) for r in valid if r.get(metric) is not None]
        if not vals:
            return
        plt.figure()
        plt.hist(vals, bins=bins, color="#4f46e5", alpha=0.8)
        plt.title(title)
        plt.xlabel(metric)
        plt.ylabel("count")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_hist.png")
        plt.close()

    _plot_hist("semantic_similarity", "Semantic similarity distribution")
    _plot_hist("word_overlap", "Word overlap distribution")
    _plot_hist("avg_relevance", "Average relevance distribution")
    _plot_hist("latency_sec", "Latency (seconds)")

    # By label bar chart if labels exist
    labels = defaultdict(int)
    for r in valid:
        labels[r.get("label", "Unknown")] += 1
    if labels:
        plt.figure()
        plt.bar(labels.keys(), labels.values(), color="#0ea5e9")
        plt.title("Samples per label")
        plt.ylabel("count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "labels.png")
        plt.close()

    print(f"Charts saved to: {output_dir}")


def run_ragas(results: List[Dict[str, Any]], output_dir: Path) -> Dict[str, float] | None:
    """Run ragas metrics on collected QA outputs."""
    if not RAGAS_AVAILABLE or ragas_evaluate is None:
        print("ragas not installed; skipping ragas evaluation")
        return None

    rows: List[Dict[str, Any]] = []
    for r in results:
        if "error" in r:
            continue
        contexts = []
        for s in r.get("sources", []):
            if isinstance(s, dict):
                contexts.append(s.get("content") or s.get("text") or "")
            else:
                contexts.append(str(s))
        rows.append(
            {
                "question": r.get("question", ""),
                "answer": r.get("model_answer", ""),
                "contexts": contexts or [""],
                "ground_truth": r.get("expected_answer", ""),
            }
        )

    if not rows:
        print("No valid rows for ragas evaluation")
        return None

    ds = Dataset.from_list(rows)
    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    print(f"Running ragas on {len(rows)} samples...")
    try:
        result = ragas_evaluate(
            dataset=ds,
            metrics=metrics,
            llm=getattr(qa_service, "_llm", None),
            embeddings=getattr(qa_service, "_embeddings", None),
        )
    except Exception as e:  # noqa: BLE001
        print(f"ragas evaluation failed: {e}")
        return None

    scores: Dict[str, float] = {}
    if hasattr(result, "scores"):
        try:
            scores = dict(result.scores)  # type: ignore[arg-type]
        except Exception:
            pass
    if hasattr(result, "to_pandas"):
        try:
            df = result.to_pandas()
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_dir / "ragas_results.csv", index=False)
            print(f"ragas per-sample results saved to: {output_dir / 'ragas_results.csv'}")
        except Exception as e:  # noqa: BLE001
            print(f"Could not save ragas results: {e}")

    if scores:
        print("ragas aggregated scores:")
        for k, v in scores.items():
            try:
                print(f"  {k}: {v:.3f}")
            except Exception:
                print(f"  {k}: {v}")
    return scores or None


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
    parser.add_argument("--visualize", action="store_true", help="Generate simple metric charts")
    parser.add_argument("--ragas", action="store_true", help="Run ragas metrics on results")
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
    visualize = args.visualize or cfg_eval.get("visualize", False)
    use_ragas = args.ragas or cfg_eval.get("use_ragas", False)

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
    if visualize:
        generate_plots(results, output_dir)
    if use_ragas:
        run_ragas(results, output_dir)

    #print("\nQUICK SUMMARY")
    #print("=" * 40)
    #print(f"Samples: {summary.get('total_samples', 0)} (valid {summary.get('valid_samples', 0)})")
    #print(f"Substring Match: {summary.get('substring_match_rate', 0):.1%}")
    #if "avg_semantic_similarity" in summary:
    #    print(f"Semantic Similarity: {summary['avg_semantic_similarity']:.3f}")
    #if summary.get("avg_llm_score") is not None:
    #    print(f"LLM Judge Score: {summary['avg_llm_score']:.3f}")
    #print(f"Avg Latency: {summary.get('avg_latency', 0):.3f}s")
    #print("Evaluation complete!\n")


    #print("\nQUICK SUMMARY (All 22 Metrics)")
    #print("=" * 60)

    summary_fields = {
        "Total samples": summary.get("total_samples"),
        "Valid samples": summary.get("valid_samples"),
        "Error samples": summary.get("error_samples"),
        "Error rate": f"{summary.get('error_rate', 0):.1%}",

        # Retrieval
        "Avg Relevance": summary.get("avg_avg_relevance"),
        "Max Relevance": summary.get("avg_max_relevance"),

        # Answer Quality
        "Substring Match Rate": f"{summary.get('substring_match_rate', 0):.1%}",
        "Avg Keyword Coverage": summary.get("avg_keyword_coverage"),
        "Avg Semantic Similarity": summary.get("avg_semantic_similarity"),
        "Avg Word Overlap": summary.get("avg_word_overlap"),
        "Avg Length Ratio": summary.get("avg_length_ratio", "N/A"),

        # Performance
        "Avg Latency (sec)": summary.get("avg_latency"),

        # LLM Judge
        "Avg LLM Overall Score": summary.get("avg_llm_score"),
        "LLM Score Std": summary.get("llm_score_std", "N/A"),
    }

    for name, val in summary_fields.items():
        print(f"{name:<30} {str(val)}")

    print("=" * 60)



if __name__ == "__main__":
    main()
