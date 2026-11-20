"""单条或批量推理；修改 main() 中的示例后直接运行。"""

import csv
from functools import lru_cache

import jieba
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from . import rf_config as cfg


@lru_cache(maxsize=1)
def load_model():
    """在进程中只加载一次模型、向量器和类别名称。"""
    model = joblib.load(cfg.MODEL_PATH)
    vectorizer = joblib.load(cfg.VECTORIZER_PATH) if isinstance(model, RandomForestClassifier) else None
    names = cfg.CLASS_PATH.read_text(encoding=cfg.ENCODING).splitlines()
    return model, vectorizer, names


def predict_many(texts):
    """将文本批量分词、向量化并返回预测类别。"""
    model, vectorizer, names = load_model()
    words = [" ".join(jieba.lcut(text)) for text in texts]
    if isinstance(model, Pipeline):
        labels = model.predict(words)
    elif isinstance(model, RandomForestClassifier):
        features = vectorizer.transform(words)
        if features.shape[1] != model.n_features_in_:
            raise ValueError("模型与词表不匹配，请重新运行 rf_train.py")
        labels = model.predict(features)
    else:
        raise TypeError("不支持的模型文件")
    return [(int(label), names[int(label)]) for label in labels]


def read_texts(path):
    """读取 TXT 每行文本，或 CSV 中的 text 列。"""
    with path.open(encoding=cfg.ENCODING, newline="") as file:
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(file)
            if "text" not in reader.fieldnames:
                raise ValueError("CSV 必须包含 text 列")
            return [row["text"] for row in reader if row["text"]]
        return [line.split("\t", 1)[0].strip() for line in file if line.strip()]


def batch(input_path, output_path):
    """批量预测并将原文、类别编号和名称保存为 CSV。"""
    texts = read_texts(input_path)
    if not texts:
        raise ValueError("输入文件没有可预测的文本")
    results = predict_many(texts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding=cfg.ENCODING, newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("text", "label", "class_name"))
        writer.writerows((text, label, name) for text, (label, name) in zip(texts, results))
    print(f"已预测 {len(texts)} 条，结果保存到 {output_path}")


def main():
    """运行下面写在代码里的单条和批量推理示例。"""
    text = "高市早苗称台湾有时就是日本有事，被我外交部严厉驳斥！"
    label, name = predict_many([text])[0]
    print(f"单条预测：{label} ({name})")

    input_file = cfg.PROCESSED_TEST_PATH
    output_file = cfg.RESULT_DIR / "batch_predictions.csv"
    batch(input_file, output_file)


if __name__ == "__main__":
    main()
