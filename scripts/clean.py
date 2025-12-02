"""
清理文档中的页眉、页码、章节标记
"""

import re
from pathlib import Path


def clean_document(text: str) -> str:
    """
    清理文档格式

    移除：
    1. 页码（单独一行的数字）
    2. 页眉（如 "Reminiscence of a Stock Operator"）
    3. 章节标记（单独的罗马数字 I, II, III）
    4. 多余的空行
    """

    lines = text.split("\n")
    cleaned_lines = []

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # 跳过空行（但保留一些空行用于段落分隔）
        if not line_stripped:
            # 避免连续多个空行
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        # 规则1: 跳过纯数字行（页码）
        if re.match(r"^\d+$", line_stripped):
            continue

        # 规则2: 跳过单个罗马数字（章节标记）
        if re.match(r"^[IVX]+$", line_stripped) and len(line_stripped) <= 5:
            continue

        # 规则3: 跳过常见的页眉模式
        if re.match(
            r"^Reminiscence[s]? of a Stock Operator$", line_stripped, re.IGNORECASE
        ):
            continue

        # 规则4: 跳过其他常见页眉格式
        headers = [
            r"^Chapter \d+$",
            r"^CHAPTER [IVX]+$",
            r"^\d+\s*$",  # 纯数字（页码变体）
        ]

        is_header = False
        for header_pattern in headers:
            if re.match(header_pattern, line_stripped, re.IGNORECASE):
                is_header = True
                break

        if is_header:
            continue

        # 保留正常内容行
        cleaned_lines.append(line)

    # 合并回文本
    cleaned_text = "\n".join(cleaned_lines)

    # 清理多余的空行（3个以上空行缩减为2个）
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def clean_file(input_path: Path, output_path: Path = None):
    """
    清理文件

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（默认为 input_cleaned.txt）
    """
    # 读取原始文件
    with open(input_path, "r", encoding="utf-8") as f:
        original_text = f.read()

    print(f"📄 原始文件: {input_path}")
    print(f"   字符数: {len(original_text)}")
    print(f"   行数: {len(original_text.split(chr(10)))}")

    # 清理
    cleaned_text = clean_document(original_text)

    print(f"\n✨ 清理后:")
    print(f"   字符数: {len(cleaned_text)}")
    print(f"   行数: {len(cleaned_text.split(chr(10)))}")

    # 保存
    if output_path is None:
        output_path = (
            input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"\n✅ 已保存到: {output_path}")

    # 显示示例
    print(f"\n📝 清理后的前500字符:")
    print("=" * 60)
    print(cleaned_text[:500])
    print("=" * 60)


def clean_directory(input_dir: Path, output_dir: Path = None):
    """
    批量清理目录下的所有文本文件
    """
    if output_dir is None:
        output_dir = input_dir / "cleaned"

    output_dir.mkdir(exist_ok=True)

    # 查找所有txt文件
    txt_files = list(input_dir.glob("*.txt"))

    print(f"找到 {len(txt_files)} 个文本文件\n")

    for txt_file in txt_files:
        print(f"处理: {txt_file.name}")
        clean_file(txt_file, output_dir / txt_file.name)
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="清理文档中的页眉和页码")
    parser.add_argument("input", type=Path, help="输入文件或目录")
    parser.add_argument("--output", "-o", type=Path, help="输出路径（可选）")
    parser.add_argument("--batch", "-b", action="store_true", help="批量处理目录")

    args = parser.parse_args()

    if args.batch:
        clean_directory(args.input, args.output)
    else:
        clean_file(args.input, args.output)
