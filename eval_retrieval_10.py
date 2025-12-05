#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用一小批人工标注的问答（当前这 10 条）来评估 RAG 系统的检索性能。

功能：
- 从命令行传入 CSV 文件路径（至少包含 question, page 字段）
- 对每个问题调用 qa_service.ask()
- 统计：
    - top-10 返回片段所在页码
    - hit@1 / hit@3 / hit@5
    - first_hit_rank
- 将结果追加列写入新的 CSV：<input>_retrieval_eval.csv
- python eval_retrieval_10.py eval/manual_10_qas.csv

"""

import argparse
import csv
import os
import re
from typing import List, Set, Tuple, Dict

import pandas as pd

from src.qa_service import qa_service


def parse_gold_pages(raw: str) -> Set[int]:
    """
    将 page 字段解析成一个页码集合：
    支持格式：
      - "76"
      - "25-26"
      - "245-246"
      - "245;246;247"
      - "245-246;300"
    """
    if raw is None:
        return set()

    s = str(raw).strip()
    if not s:
        return set()

    pages: Set[int] = set()
    # 先按 ; 或 , 拆分多个段
    for part in re.split(r"[;,]", s):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            # 范围 "25-26"
            start_str, end_str = part.split("-", 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for p in range(start, end + 1):
                pages.add(p)
        else:
            # 单页
            try:
                pages.add(int(part))
            except ValueError:
                continue
    return pages


def compute_hits(
    gold_pages: Set[int],
    retrieved_pages: List[int],
    ks: Tuple[int, ...] = (1, 3, 5),
) -> Tuple[Dict[int, bool], int | None]:
    """
    计算 hit@k 和 first_hit_rank。
    - gold_pages: 黄金页码集合
    - retrieved_pages: 按顺序返回的页码列表（长度 <= top_k）
    返回：
      - hits: {1: bool, 3: bool, 5: bool, ...}
      - first_hit_rank: 第一次命中的 rank（从 1 开始），如果完全没命中则为 None
    """
    first_hit_rank = None
    for rank, p in enumerate(retrieved_pages, start=1):
        if p in gold_pages:
            first_hit_rank = rank
            break

    hits: Dict[int, bool] = {}
    for k in ks:
        hits[k] = first_hit_rank is not None and first_hit_rank <= k

    return hits, first_hit_rank


def retrieve_pages_for_question(
    question: str,
    top_k: int = 10,
) -> List[int]:
    """
    调用 qa_service.ask(question)，拿出前 top_k 个 source 的 page 字段。
    注意：这里评估的是“整个 RAG 管线里当前的检索 + 重排效果”，
    而不是单独某个 retriever。
    """
    out = qa_service.ask(question)
    sources = out.get("sources", []) or []

    pages: List[int] = []
    for doc in sources[:top_k]:
        # 兼容不同字段写法
        page = None
        if isinstance(doc, dict):
            page = doc.get("page") or doc.get("metadata", {}).get("page")
        else:
            # 兼容 LangChain Document 对象
            metadata = getattr(doc, "metadata", {}) or {}
            page = getattr(doc, "page", None) or metadata.get("page")

        if page is not None:
            try:
                pages.append(int(page))
            except (TypeError, ValueError):
                continue

    return pages


def _normalize_text(s: str) -> str:
    """用于判断是否‘原文’，简单归一化一下大小写和空白。"""
    if s is None:
        return ""
    s = str(s)
    # 去掉首尾引号
    s = s.strip().strip('"').strip("'")
    # 小写 + 合并空白
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _infer_qa_type(answer: str, context: str) -> str:
    """
    粗略区分：
      - verbatim: answer 基本是 context 的原文（或 context 是 answer 的子串）
      - summary: 其余情况
      - unknown: 缺字段/太短
    """
    a = _normalize_text(answer)
    c = _normalize_text(context)

    if not a or not c:
        return "unknown"

    # 太短的句子也很难判断，直接给 unknown
    if len(a.split()) <= 3:
        return "unknown"

    if a in c or c in a:
        return "verbatim"
    return "summary"


def evaluate_file(
    csv_path: str,
    top_k: int = 10,
) -> str:
    """
    对指定 CSV 文件中的所有问题进行检索评估。

    返回：输出文件路径。
    """
    df = pd.read_csv(csv_path)

    if "question" not in df.columns or "page" not in df.columns:
        raise ValueError("CSV 必须包含 'question' 和 'page' 两列")

    # 如果有 answer + context，就顺便打一个类型标签：verbatim / summary
    if "answer" in df.columns and "context" in df.columns:
        qa_types: List[str] = []
        for _, row in df.iterrows():
            qa_types.append(_infer_qa_type(row["answer"], row["context"]))
        df["qa_type"] = qa_types

    # 初始化一次即可
    qa_service.initialize()

    hit1_list: List[bool] = []
    hit3_list: List[bool] = []
    hit5_list: List[bool] = []
    first_rank_list: List[int | None] = []
    retrieved_list: List[str] = []

    for idx, row in df.iterrows():
        q = str(row["question"])
        gold_raw = row["page"]
        gold_pages = parse_gold_pages(gold_raw)

        print(f"[{idx + 1}/{len(df)}] Q: {q[:80]}...  gold_pages={sorted(gold_pages)}")

        retrieved_pages = retrieve_pages_for_question(q, top_k=top_k)

        hits, first_rank = compute_hits(gold_pages, retrieved_pages, ks=(1, 3, 5))

        # 保存结果
        hit1_list.append(hits[1])
        hit3_list.append(hits[3])
        hit5_list.append(hits[5])
        first_rank_list.append(first_rank if first_rank is not None else -1)
        retrieved_list.append(";".join(str(p) for p in retrieved_pages))

    df["hit_at_1"] = hit1_list
    df["hit_at_3"] = hit3_list
    df["hit_at_5"] = hit5_list
    df["first_hit_rank"] = first_rank_list  # -1 表示完全没命中
    df["retrieved_pages_top10"] = retrieved_list

    base, ext = os.path.splitext(csv_path)
    out_path = f"{base}_retrieval_eval{ext}"
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\n✅ 评估完成，结果已写入: {out_path}")

    # 打印整体指标
    total = len(df)
    hit1_rate = sum(hit1_list) / total
    hit3_rate = sum(hit3_list) / total
    hit5_rate = sum(hit5_list) / total
    print(f"\nSummary on {total} samples:")
    print(f"  hit@1 = {hit1_rate:.3f}")
    print(f"  hit@3 = {hit3_rate:.3f}")
    print(f"  hit@5 = {hit5_rate:.3f}")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="用 10 条高质量样本测试当前 RAG 系统的检索效果（页码命中情况）"
    )
    parser.add_argument(
        "csv",
        help="输入 CSV 文件路径（至少包含 question, page 两列）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="每个问题保留的检索结果数（默认 10）",
    )

    args = parser.parse_args()
    evaluate_file(args.csv, top_k=args.top_k)


if __name__ == "__main__":
    main()
