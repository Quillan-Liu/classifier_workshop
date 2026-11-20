"""命令行入口：

    python -m train.dataset --input <数据集> --use bert|fasttext|rf|all --out <输出目录>
"""

from ..logging import log_call as _log_call

import argparse
import json
from pathlib import Path

from .config import DEFAULT_DEV_RATIO, DEFAULT_SEED, DEFAULT_TEST_RATIO, PURPOSES
from .pipeline import prepare


def _parse_labels(value):
    """将逗号分隔的类名或每行一个类名的文件转换成列表。

    Args:
        value: 待校验或转换的输入值。

    Returns:
        去除空项后的类名列表；未提供输入时返回 None。
    """
    with _log_call("train.dataset.__main__._parse_labels", locals()):
        if not value:
            return None
        path = Path(value)
        if path.is_file():
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [item.strip() for item in value.split(",") if item.strip()]


def build_parser():
    """构建数据集处理命令行参数，包括输入、导出类型和划分设置。

    Returns:
        argparse.ArgumentParser 实例。
    """
    with _log_call("train.dataset.__main__.build_parser", locals()):
        parser = argparse.ArgumentParser(
            prog="python -m train.dataset",
            description="把数据集处理成可直接训练的产物（bert / fasttext / rf）",
        )
        parser.add_argument("--input", required=True, help="数据集文件或目录（.arrow/.jsonl/txt）")
        parser.add_argument("--use", required=True, choices=[*PURPOSES, "all"], help="用途")
        parser.add_argument("--out", default=None, help="输出根目录，默认 data/prepared/<数据集名>")
        parser.add_argument("--labels", default=None, help="类别名，逗号分隔或 class.txt 路径")
        parser.add_argument("--dev-ratio", type=float, default=DEFAULT_DEV_RATIO, help="从 train 切 dev 的比例")
        parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="无 test 时从 train 切的比例")
        parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="切分随机种子")
        parser.add_argument("--stopwords", default=None, help="RF 的停用词表路径")
        parser.add_argument("--allow-conflicts", action="store_true", help="允许同一文本多个标签（保留首次出现）")
        return parser


def main(argv=None):
    """解析命令行参数并执行数据集准备和文件导出。

    Args:
        argv: 命令行参数列表；None 表示读取当前进程参数。

    Returns:
        各分组样本数的 JSON 字符串。
    """
    with _log_call("train.dataset.__main__.main", locals()):
        args = build_parser().parse_args(argv)
        manifest = prepare(
            input_path=args.input,
            use=args.use,
            out_dir=args.out,
            labels=_parse_labels(args.labels),
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            stopwords=args.stopwords,
            allow_conflicts=args.allow_conflicts,
        )
        return json.dumps(manifest["rows"], ensure_ascii=False)


if __name__ == "__main__":
    main()
