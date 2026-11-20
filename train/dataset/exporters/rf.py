"""RF 用途：产出 jieba 分词 CSV + TF-IDF 稀疏矩阵，直接对接 rf_train.py。"""

from ...logging import log_call as _log_call

from ..config import DEFAULT_STOPWORDS, ENCODING


def _load_stopwords(path):
    """读取指定或默认停用词文件，忽略空行及含空白的词条。

    Args:
        path: 待读取的数据文件、目录或配置文件路径。

    Returns:
        停用词列表；没有可用文件配置或词条时返回 None。
    """
    with _log_call("train.dataset.exporters.rf._load_stopwords", locals()):
        path = path or (DEFAULT_STOPWORDS if DEFAULT_STOPWORDS.is_file() else None)
        if not path:
            return None
        words = [line.strip() for line in open(path, encoding=ENCODING)]
        words = [w for w in words if w and not any(ch.isspace() for ch in w)]
        return words or None


def export(out_dir, splits, classes, stopwords=None, seed=42, tfidf_max_features=None, **_):
    """导出分词 CSV，仅用训练集拟合 TF-IDF，并保存各分组特征、标签和向量器。

    Args:
        out_dir: 导出文件的目标目录；None 时按数据集名称生成默认目录。
        splits: 包含 train、dev、test 等分组的样本字典。
        classes: 类别编号到名称的字典，报告排版时可为有序类名列表。
        stopwords: RF 停用词文件路径；None 时尝试默认停用词文件。
        seed: 用于数据划分或训练的随机种子。
        tfidf_max_features: TF-IDF 最大特征数；None 表示不限制。
        _: 为兼容统一导出器调用而接收的额外关键字参数，本导出器忽略。

    Returns:
        包含 dir、各分组 features（shape/nnz）、vocab_size 和 note 的字典。
    """
    with _log_call("train.dataset.exporters.rf.export", locals()):
        import csv

        import jieba
        import joblib
        import numpy as np
        from scipy.sparse import save_npz
        from sklearn.feature_extraction.text import TfidfVectorizer

        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) 分词 CSV，列与 rf_preprocess 保持一致
        words = {}
        for split, samples in splits.items():
            path = out_dir / f"process_{split}.csv"
            with path.open("w", encoding=ENCODING, newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("text", "label", "words"))
                rows = []
                for text, label in samples:
                    tokenized = " ".join(jieba.lcut(text))
                    writer.writerow([text, label, tokenized])
                    rows.append(tokenized)
            words[split] = rows

        # 2) 在 train 上拟合 TF-IDF，再转换 dev/test
        vectorizer = TfidfVectorizer(
            tokenizer=str.split, token_pattern=None, lowercase=False,
            stop_words=_load_stopwords(stopwords), dtype=np.float32,
            max_features=tfidf_max_features,
        )
        shapes = {}
        matrix = vectorizer.fit_transform(words["train"])
        for split, rows in words.items():
            data = matrix if split == "train" else vectorizer.transform(rows)
            data = data.tocsr()
            save_npz(out_dir / f"{split}_tfidf.npz", data)
            np.save(out_dir / f"{split}_labels.npy", np.array([label for _, label in splits[split]]))
            shapes[split] = {"shape": list(data.shape), "nnz": int(data.nnz)}

        joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")
        with (out_dir / "vocabulary.tsv").open("w", encoding=ENCODING, newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(("word_index", "word"))
            writer.writerows(enumerate(vectorizer.get_feature_names_out()))
        # 类名落一份，评估时能把数字标签还原成可读名字
        (out_dir / "class.txt").write_text(
            "\n".join(classes[index] for index in sorted(classes)) + "\n", encoding=ENCODING)

        return {
            "dir": str(out_dir),
            "features": shapes,
            "vocab_size": len(vectorizer.vocabulary_),
            "note": "rf_config 里把 PROCESSED_DATA_DIR / MODEL_DIR 指向该目录即可训练",
        }
