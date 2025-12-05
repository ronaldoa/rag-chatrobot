#!/usr/bin/env python

# 最常见用法
#python split_llm_cases.py eval/test_detail_result.csv

# 指定输出目录
#python split_llm_cases.py eval/test_detail_result.csv --out-dir #eval/timing_splits

# 调整阈值（比如高分更严格）
#python split_llm_cases.py eval/test_detail_result.csv --high-thr 0.85 --low-thr 0.25

import argparse
from pathlib import Path

import pandas as pd


def categorize_row(row, high_thr, low_thr):
    overall = float(row.get("llm_overall", 0.0))
    acc = float(row.get("llm_accuracy", 0.0))
    faith = float(row.get("llm_faithfulness", 0.0))

    # 高分：整体分高，准确 & faithfulness 都不错
    if (overall >= high_thr) and (acc >= high_thr * 0.95) and (faith >= high_thr * 0.9):
        return "high"

    # 低分：整体或其中任一很低
    if (overall <= low_thr) or (acc <= low_thr) or (faith <= low_thr):
        return "low"

    # 其余是边界样本
    return "border"


def main():
    parser = argparse.ArgumentParser(
        description="Split LLM eval CSV into high / low / border subsets."
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Input CSV file, e.g. eval/test_detail_result.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="eval_results",
        help="Output directory (default: eval_results)",
    )
    parser.add_argument(
        "--high-thr",
        type=float,
        default=0.80,
        help="Threshold for high-score samples on llm_overall (default: 0.80)",
    )
    parser.add_argument(
        "--low-thr",
        type=float,
        default=0.30,
        help="Threshold for low-score samples on llm_overall / accuracy / faithfulness (default: 0.30)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    # 打标签
    df["llm_category"] = df.apply(
        lambda r: categorize_row(r, args.high_thr, args.low_thr), axis=1
    )

    df_high = df[df["llm_category"] == "high"]
    df_low = df[df["llm_category"] == "low"]
    df_border = df[df["llm_category"] == "border"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem  # 比如 test_detail_result
    df_high.to_csv(out_dir / f"{stem}_high_cases.csv", index=False)
    df_low.to_csv(out_dir / f"{stem}_low_cases.csv", index=False)
    df_border.to_csv(out_dir / f"{stem}_border_cases.csv", index=False)

    print(f"Input file: {input_path}")
    print(f"High samples:      {len(df_high)}  -> {out_dir / f'{stem}_high_cases.csv'}")
    print(f"Low samples:       {len(df_low)}  -> {out_dir / f'{stem}_low_cases.csv'}")
    print(f"Borderline samples:{len(df_border)}  -> {out_dir / f'{stem}_border_cases.csv'}")


if __name__ == "__main__":
    main()

