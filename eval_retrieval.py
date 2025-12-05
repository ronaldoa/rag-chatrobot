#!/usr/bin/env python
"""
评估检索质量的脚本（不经过 LLM，只看 top-K 里有没有命中“金标准页码/来源”）

用法示例：

    python eval_retrieval.py tests/timing_test.csv \
        --question-col question \
        --page-col page \
        --source-col source \
        --top-k 1 3 5

脚本会输出：
- 命中率统计 (hit@K)
- 带有每条样本 hit 情况的 csv：<输入名>_retrieval_eval.csv
-python eval_retrieval.py tests/timing_test.csv --question-col question --page-col page --top-k 1 3 5

"""

from __future__ import annotations

import argparse
import os
from typing import List, Dict, Any

import pandas as pd

from src.qa_service import qa_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retriever hit@K metrics.")
    parser.add_argument(
        "csv_path",
        type=str,
        help="输入测试集 CSV 文件路径（必须至少包含 question + gold page 列）",
    )
    parser.add_argument(
        "--question-col",
        type=str,
        default="question",
        help="问题列名（默认: question）",
    )
    parser.add_argument(
        "--page-col",
        type=str,
        default="page",
        help="金标准页码列名（默认: page）",
    )
    parser.add_argument(
        "--source-col",
        type=str,
        default=None,
        help="金标准来源列名（可选，比如 pdf 文件名 / 书名）。如果为 None 则只按页码匹配。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="需要统计的 K 值列表，例如: --top-k 1 3 5 10（默认: 1 3 5）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多评估多少条样本（调试用，可不填）。",
    )
    return parser.parse_args()


def normalize_gold_pages(raw: Any) -> List[int]:
    """
    支持几种形式：
    - 单个数字: 12
    - 字符串: "12" / "12,13" / "12;13" / "12|13"
    返回：整数列表
    """
    if pd.isna(raw):
        return []

    if isinstance(raw, (int, float)):
        return [int(raw)]

    s = str(raw).strip()
    if not s:
        return []

    # 常见分隔符
    for sep in [";", ",", "|", "/"]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            return [int(p) for p in parts if p.isdigit()]

    # 纯数字
    if s.isdigit():
        return [int(s)]

    # 其他奇怪格式尽量 parse 一下
    digits = "".join(ch if ch.isdigit() else " " for ch in s).split()
    return [int(d) for d in digits] if digits else []


def get_doc_source_page(doc: Dict[str, Any]) -> tuple[str | None, int | None]:
    """
    从 qa_service 返回的 source dict 中安全提取 (source, page)

    你在 debug_one 里是 doc.get("source"), doc.get("page"),
    这里沿用这个约定。
    """
    src = doc.get("source")
    page = doc.get("page")

    # 有些实现可能把 page 放在 metadata 里，做个兼容
    if page is None:
        meta = doc.get("metadata") or {}
        page = meta.get("page")

    try:
        page_int = int(page) if page is not None else None
    except Exception:
        page_int = None

    return src, page_int


def evaluate_retrieval(
    csv_path: str,
    question_col: str = "question",
    page_col: str = "page",
    source_col: str | None = None,
    top_k_list: List[int] = [1, 3, 5],
    max_samples: int | None = None,
) -> None:
    # 1) 读 CSV
    df = pd.read_csv(csv_path)
    if question_col not in df.columns:
        raise ValueError(f"找不到问题列: {question_col}")
    if page_col not in df.columns:
        raise ValueError(f"找不到页码列: {page_col}")
    if source_col is not None and source_col not in df.columns:
        raise ValueError(f"找不到来源列: {source_col}")

    if max_samples is not None:
        df = df.head(max_samples)

    # 2) 初始化 QA service（内部会初始化 retriever / vector store）
    qa_service.initialize()

    # 3) 为每个样本评估 hit@K
    top_k_list = sorted(set(top_k_list))
    total = len(df)
    hit_counts = {k: 0 for k in top_k_list}
    first_hit_ranks: List[int | None] = []

    # 新增列：记录每条样本的命中情况
    per_sample_hits = {k: [] for k in top_k_list}
    per_sample_first_hit_rank: List[int | None] = []
    per_sample_retrieved_pages: List[str] = []
    per_sample_retrieved_sources: List[str] = []

    for idx, row in df.iterrows():
        question = str(row[question_col])
        gold_pages = normalize_gold_pages(row[page_col])
        gold_source = str(row[source_col]) if source_col is not None else None

        # 允许 gold_pages 为空，但这类样本会视作“无 gold”，hit 永远为 False
        # 你可以根据需要过滤掉。
        result = qa_service.ask(question)
        sources = result.get("sources", []) or []

        # 记录检索到的前若干个文档的 (source, page)
        retrieved_info = []
        for doc in sources:
            src, page = get_doc_source_page(doc)
            retrieved_info.append((src, page))

        # 方便排查：把前 10 个的 (src, page) 存成字符串
        per_sample_retrieved_pages.append(
            ";".join(str(p) for (_, p) in retrieved_info[:10])
        )
        per_sample_retrieved_sources.append(
            ";".join(str(s) for (s, _) in retrieved_info[:10])
        )

        # 计算 rank hit
        first_rank = None

        for rank, (src, page) in enumerate(retrieved_info, start=1):
            # page 匹配
            page_match = page in gold_pages if page is not None else False

            # 如果指定了 gold_source，则要求 source 也匹配；否则只看 page
            if source_col is not None:
                src_match = (src == gold_source) if gold_source is not None else False
                is_hit = page_match and src_match
            else:
                is_hit = page_match

            if is_hit:
                first_rank = rank
                # 更新各个 K 的命中
                for k in top_k_list:
                    if rank <= k:
                        hit_counts[k] += 1
                        per_sample_hits[k].append(True)
                    else:
                        per_sample_hits[k].append(False)
                break

        # 如果没有任何命中
        if first_rank is None:
            for k in top_k_list:
                per_sample_hits[k].append(False)

        per_sample_first_hit_rank.append(first_rank)

        print(
            f"[{idx + 1}/{total}] "
            f"Q: {question[:40]}... | gold_pages={gold_pages} "
            f"| first_hit_rank={first_rank}"
        )

    # 4) 统计总体 hit@K
    print("\n===== Retrieval Hit@K Summary =====")
    for k in top_k_list:
        hit_rate = hit_counts[k] / total if total > 0 else 0.0
        print(f"Hit@{k}: {hit_counts[k]} / {total} = {hit_rate:.3f}")

    # 5) 写回结果到新的 CSV
    out_df = df.copy()
    for k in top_k_list:
        out_df[f"hit_at_{k}"] = per_sample_hits[k]
    out_df["first_hit_rank"] = per_sample_first_hit_rank
    out_df["retrieved_pages_top10"] = per_sample_retrieved_pages
    out_df["retrieved_sources_top10"] = per_sample_retrieved_sources

    base, ext = os.path.splitext(csv_path)
    out_path = f"{base}_retrieval_eval{ext or '.csv'}"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n详细结果已保存到: {out_path}")


def main():
    args = parse_args()
    evaluate_retrieval(
        csv_path=args.csv_path,
        question_col=args.question_col,
        page_col=args.page_col,
        source_col=args.source_col,
        top_k_list=args.top_k,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
