"""fastText 用途：按所选字符级或词级产出 `__label__<类名> 空格分词` 的 txt。"""

from ...logging import log_call as _log_call

from ..config import ENCODING, FASTTEXT_PUNCT


def _char_text(text):
    """移除标点和空白，将剩余字符用空格连接为字符级 token。

    Args:
        text: 待清洗或分词的原始文本。

    Returns:
        字符级分词字符串。
    """
    return " ".join(char for char in text if char not in FASTTEXT_PUNCT and not char.isspace())


def _word_text(text):
    """使用 jieba 分词，移除空 token 和配置中的标点。

    Args:
        text: 待清洗或分词的原始文本。

    Returns:
        以空格分隔的词级分词字符串。
    """
    import jieba

    words = (word.strip() for word in jieba.lcut(text))
    return " ".join(word for word in words if word and word not in FASTTEXT_PUNCT)


def export(out_dir, splits, classes, word_level=False, **_):
    """按所选粒度写入 fastText 各分组训练文件和类别文件。

    Args:
        out_dir: 导出文件的目标目录；None 时按数据集名称生成默认目录。
        splits: 包含 train、dev、test 等分组的样本字典。
        classes: 类别编号到名称的字典，报告排版时可为有序类名列表。
        word_level: True 使用 jieba 词级分词；False 使用字符级拆分。
        _: 为兼容统一导出器调用而接收的额外关键字参数，本导出器忽略。

    Returns:
        包含输出目录、文件样本数及使用说明的字典。
    """
    with _log_call("train.dataset.exporters.fasttext.export", locals()):
        out_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        tokenizer, transform = ("word", _word_text) if word_level else ("char", _char_text)
        for split, samples in splits.items():
            path = out_dir / f"{split}_fastText_{tokenizer}.txt"
            with path.open("w", encoding=ENCODING, newline="\n") as file:
                for text, label in samples:
                    file.write(f"__label__{classes[label]} {transform(text)}\n")
            files[path.name] = len(samples)

        (out_dir / "class.txt").write_text(
            "\n".join(classes[index] for index in sorted(classes)) + "\n", encoding=ENCODING)

        return {
            "dir": str(out_dir),
            "files": files,
            "note": "fasttext.train_supervised(input=<对应 split 文件>, label='__label__')",
        }
