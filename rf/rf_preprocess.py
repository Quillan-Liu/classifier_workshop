"""将原始新闻文本分词，生成训练、验证和测试 CSV。"""

import csv
import jieba
from .rf_config import ENCODING, OUTPUT_COLUMNS, PROCESSED_DATA_DIR, RAW_DATA_DIR, RAW_DELIMITER, SPLITS


def process(split):
    """读取一个原始数据集，分词后写成 CSV。"""
    source = RAW_DATA_DIR / f"{split}.txt"
    target = PROCESSED_DATA_DIR / f"process_{split}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("r", encoding=ENCODING, newline="") as src, target.open(
        "w", encoding=ENCODING, newline=""
    ) as dst:
        reader = csv.reader(src, delimiter=RAW_DELIMITER)
        writer = csv.writer(dst)
        writer.writerow(OUTPUT_COLUMNS)
        for count, row in enumerate(reader, 1):
            if len(row) != 2 or not row[0] or not row[1].isdigit():
                raise ValueError(f"{source}:{count}: 无效数据行 {row!r}")
            text, label = row
            writer.writerow([text, label, " ".join(jieba.lcut(text))])
    print(f"{split}: {count} 条 -> {target}")


if __name__ == "__main__":
    for name in SPLITS:
        process(name)
