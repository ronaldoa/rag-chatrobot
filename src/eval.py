"""
Lightweight RAG evaluation runner - 支持CSV格式

Usage:
    python -m src.eval --eval-path eval.csv --out results.csv --limit 100

Input file (CSV):
    Question,Answer,Label
    "What was...", "He noticed that...", "Timing"
"""
import argparse
import csv
import time
from pathlib import Path
from typing import List, Dict, Any

from .qa_service import qa_service


def load_eval_set_csv(path: Path) -> List[Dict[str, Any]]:
    """
    从CSV文件读取评估数据

    CSV格式:
        Question,Answer,Label
        "问题1", "答案1", "类别1"
        "问题2", "答案2", "类别2"
    """
    samples: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader, start=1):
            # 获取字段（兼容不同的列名）
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
                print(f"⚠️  跳过第{row_idx}行：问题为空")
                continue

            samples.append({
                "question": question,
                "answer": answer,      # 作为期望答案
                "label": label,        # 保留标签信息
                "row_idx": row_idx
            })

    return samples


def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    """自动检测文件格式并加载"""
    if path.suffix.lower() == '.csv':
        return load_eval_set_csv(path)
    elif path.suffix.lower() == '.jsonl':
        # 原有的JSONL加载逻辑
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
        raise ValueError(f"不支持的文件格式: {path.suffix}，请使用.csv或.jsonl")


def evaluate(samples: List[Dict[str, Any]], limit: int | None = None) -> List[Dict[str, Any]]:
    """运行评估"""
    qa_service.initialize()
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

            # 评估指标
            # 1. 简单子串匹配
            substring_match = bool(expected) and (expected.lower() in answer.lower())

            # 2. 更精确的匹配：计算关键词覆盖率
            keyword_coverage = calculate_keyword_coverage(expected, answer)

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
                "latency_sec": round(latency, 3),
                "num_sources": len(sources),
                "sources": [s.get("source", "") for s in sources]
            })

            print(f"  ✓ 匹配: {substring_match} | 关键词: {keyword_coverage:.2f} | 延迟: {latency:.2f}s")

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
                "latency_sec": round(latency, 3),
                "num_sources": 0,
                "sources": []
            })
            print(f"  ❌ 错误: {e}")

    return results


def calculate_keyword_coverage(expected: str, answer: str) -> float:
    """
    计算期望答案的关键词在生成答案中的覆盖率

    Returns:
        0.0-1.0的分数
    """
    if not expected or not answer:
        return 0.0

    # 简单分词（可以用更复杂的分词器）
    import re

    # 提取期望答案的关键词（去除常见停用词）
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'he', 'she', 'it', 'they', 'his', 'her', 'their', 'this', 'that',
        '的', '了', '是', '在', '有', '和', '与', '或', '但'
    }

    expected_words = set(re.findall(r'\w+', expected.lower()))
    expected_words = {w for w in expected_words if w not in stopwords and len(w) > 2}

    if not expected_words:
        return 0.0

    answer_lower = answer.lower()

    # 计算覆盖的关键词数
    covered = sum(1 for word in expected_words if word in answer_lower)

    return covered / len(expected_words)


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """写入CSV结果"""
    fieldnames = [
        "idx", "row_idx", "label", "question", "expected", "answer",
        "substring_match", "keyword_coverage", "latency_sec",
        "num_sources", "sources"
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # sources列表转为字符串
            if isinstance(row.get("sources"), list):
                row["sources"] = "; ".join(row["sources"])
            writer.writerow(row)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总统计"""
    total = len(rows)
    if total == 0:
        return {}

    # 基础统计
    substring_matches = sum(1 for r in rows if r.get("substring_match"))
    avg_keyword_coverage = sum(r.get("keyword_coverage", 0.0) for r in rows) / total
    avg_latency = sum(r.get("latency_sec", 0.0) for r in rows) / total

    # 按Label分组统计
    label_stats = {}
    for row in rows:
        label = row.get("label", "Unknown")
        if label not in label_stats:
            label_stats[label] = {
                "count": 0,
                "substring_matches": 0,
                "avg_coverage": 0.0,
                "avg_latency": 0.0
            }

        label_stats[label]["count"] += 1
        if row.get("substring_match"):
            label_stats[label]["substring_matches"] += 1
        label_stats[label]["avg_coverage"] += row.get("keyword_coverage", 0.0)
        label_stats[label]["avg_latency"] += row.get("latency_sec", 0.0)

    # 计算平均值
    for label, stats in label_stats.items():
        count = stats["count"]
        stats["match_rate"] = round(stats["substring_matches"] / count, 3)
        stats["avg_coverage"] = round(stats["avg_coverage"] / count, 3)
        stats["avg_latency"] = round(stats["avg_latency"] / count, 3)

    return {
        "total": total,
        "substring_matches": substring_matches,
        "substring_match_rate": round(substring_matches / total, 3),
        "avg_keyword_coverage": round(avg_keyword_coverage, 3),
        "avg_latency_sec": round(avg_latency, 3),
        "by_label": label_stats
    }


def main():
    parser = argparse.ArgumentParser(description="评估RAG系统（支持CSV和JSONL格式）")
    parser.add_argument("--eval-path", type=Path, required=True,
                       help="评估数据路径（.csv或.jsonl）")
    parser.add_argument("--out", type=Path, default=Path("eval_results.csv"),
                       help="结果保存路径")
    parser.add_argument("--limit", type=int, default=None,
                       help="限制评估的样本数量")
    args = parser.parse_args()

    print(f"\n📂 加载数据: {args.eval_path}")
    samples = load_eval_set(args.eval_path)
    print(f"✓ 加载了 {len(samples)} 个问题\n")

    print("🚀 开始评估...\n")
    rows = evaluate(samples, limit=args.limit)

    print("\n📊 生成统计...")
    stats = summarize(rows)

    print("\n" + "="*60)
    print("📈 评估结果汇总")
    print("="*60)
    print(f"总样本数:     {stats['total']}")
    print(f"子串匹配率:   {stats['substring_match_rate']:.1%}")
    print(f"关键词覆盖:   {stats['avg_keyword_coverage']:.1%}")
    print(f"平均延迟:     {stats['avg_latency_sec']:.3f}s")

    if stats.get('by_label'):
        print("\n按标签分类:")
        for label, label_stats in stats['by_label'].items():
            print(f"\n  【{label}】")
            print(f"    样本数:   {label_stats['count']}")
            print(f"    匹配率:   {label_stats['match_rate']:.1%}")
            print(f"    覆盖率:   {label_stats['avg_coverage']:.1%}")
            print(f"    平均延迟: {label_stats['avg_latency']:.3f}s")

    print("="*60 + "\n")

    write_csv(rows, args.out)
    print(f"✅ 结果已保存到: {args.out}\n")


if __name__ == "__main__":
    main()