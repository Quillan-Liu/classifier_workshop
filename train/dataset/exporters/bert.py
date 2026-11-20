"""BERT 用途：产出项目 bert/src 那套 `text<TAB>label` 的 TSV + class.txt。"""

from ...logging import log_call as _log_call

from ..config import ENCODING


def export(out_dir, splits, classes, **_):
    """创建输出目录，写入各分组的 text<TAB>label 文件及 class.txt。

    Args:
        out_dir: 导出文件的目标目录；None 时按数据集名称生成默认目录。
        splits: 包含 train、dev、test 等分组的样本字典。
        classes: 类别编号到名称的字典，报告排版时可为有序类名列表。
        _: 为兼容统一导出器调用而接收的额外关键字参数，本导出器忽略。

    Returns:
        包含输出目录、文件样本数及使用说明的字典。
    """
    with _log_call("train.dataset.exporters.bert.export", locals()):
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = {}
        for split, samples in splits.items():
            path = out_dir / f"{split}.txt"
            with path.open("w", encoding=ENCODING, newline="\n") as file:
                for text, label in samples:
                    file.write(f"{text}\t{label}\n")
            rows[split] = len(samples)

        class_path = out_dir / "class.txt"
        class_path.write_text("\n".join(classes[index] for index in sorted(classes)) + "\n",
                              encoding=ENCODING)

        return {
            "dir": str(out_dir),
            "files": {f"{name}.txt": count for name, count in rows.items()} | {"class.txt": len(classes)},
            "note": "把 bert_config 的 data_path 指向该目录，num_classes 会自动等于 len(class.txt)",
        }
