#!/usr/bin/env python
"""
Step 1: 调用本地 RAG 系统，生成 predictions.jsonl

用法:
    python scripts/run_rag_predictions.py \
        --eval-path eval/test.csv \
        --output eval/predictions.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

# ===== 关键：把项目根目录加入 sys.path =====
import sys

ROOT = Path(__file__).resolve().parents[1]  # /.../llama3-chatbot-hybrid
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qa_service import qa_service  # 现在就能正常 import 了


def load_eval_csv(path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("Question") or row.get("question") or "").strip()
            a = (row.get("Answer") or row.get("answer") or "").strip()
            label = (row.get("Label") or row.get("label") or "").strip()
            if not q:
                continue
            samples.append(
                {
                    "question": q,
                    "ground_truth": a,
                    "label": label,
                }
            )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-path",
        type=Path,
        required=True,
        help="CSV file with Question / Answer / Label",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.jsonl"),
        help="Where to save predictions JSONL",
    )
    args = parser.parse_args()

    samples = load_eval_csv(args.eval_path)
    print(f"Loaded {len(samples)} samples from {args.eval_path}")

    qa_service.initialize()  # ✅ 只在这个脚本里初始化 LLM + 向量库

    out_lines: List[str] = []

    for idx, s in enumerate(samples, start=1):
        q = s["question"]
        gt = s["ground_truth"]
        label = s["label"]

        preview = q[:60] + "..." if len(q) > 60 else q
        print(f"[{idx}/{len(samples)}] {preview}")

        res = qa_service.ask(q)

        answer = res.get("answer", "")
        sources = res.get("sources", [])

        # 把 sources 转成纯文本列表，方便 ragas 使用
        ctx_texts: List[str] = []
        for src in sources:
            if isinstance(src, dict):
                text = (
                    src.get("text")
                    or src.get("content")
                    or src.get("page_content")
                    or ""
                )
                if text:
                    ctx_texts.append(str(text))
            else:
                if src:
                    ctx_texts.append(str(src))

        if not ctx_texts:
            ctx_texts = [""]

        record = {
            "question": q,
            "answer": answer,
            "contexts": ctx_texts,
            "ground_truth": gt,
            "label": label,
        }

        out_lines.append(json.dumps(record, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nSaved predictions to: {args.output}")


if __name__ == "__main__":
    main()

