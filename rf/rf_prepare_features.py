"""把分词后的句子转为 CSR：行=样本，列=词下标，值=TF-IDF。"""

import csv

import joblib
import numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from .rf_config import (
    CSR_PATHS, ENCODING, LABEL_PATHS, PROCESSED_DEV_PATH, PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH, STOPWORDS_PATH, TFIDF_MAX_FEATURES, VECTORIZER_PATH, VOCAB_PATH,
)


def load_rows(path):
    """读取分词 CSV，返回句子词串和对应标签。"""
    with path.open(encoding=ENCODING, newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or any(not row["words"] or not row["label"] for row in rows):
        raise ValueError(f"数据为空或存在缺失值: {path}")
    return [row["words"] for row in rows], np.array([int(row["label"]) for row in rows])


def make_vectorizer():
    """创建按空格分词、使用停用词表的 TF-IDF 向量器。"""
    stopwords = [word for word in STOPWORDS_PATH.read_text(encoding=ENCODING).splitlines()
                 if word and not any(char.isspace() for char in word)]
    return TfidfVectorizer(
        tokenizer=str.split, token_pattern=None, lowercase=False,
        stop_words=stopwords, max_features=TFIDF_MAX_FEATURES, dtype=np.float32,
    )


def save_split(name, path, vectorizer, fit=False):
    """将一个数据集转换为 CSR，并分别保存特征和标签。"""
    texts, labels = load_rows(path)
    matrix = vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)
    matrix = matrix.tocsr()
    save_npz(CSR_PATHS[name], matrix)
    np.save(LABEL_PATHS[name], labels)
    print(f"{name}: shape={matrix.shape}, nnz={matrix.nnz}, format={matrix.getformat()}")


def main():
    """在训练集上拟合词表，再转换验证集和测试集。"""
    VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    vectorizer = make_vectorizer()
    save_split("train", PROCESSED_TRAIN_PATH, vectorizer, fit=True)
    for name, path in (("dev", PROCESSED_DEV_PATH), ("test", PROCESSED_TEST_PATH)):
        save_split(name, path, vectorizer)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    with VOCAB_PATH.open("w", encoding=ENCODING, newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(("word_index", "word"))
        writer.writerows(enumerate(vectorizer.get_feature_names_out()))
    print(f"词表已保存: {VOCAB_PATH}")


if __name__ == "__main__":
    main()
