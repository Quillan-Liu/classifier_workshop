"""使用登记模型对输入字符串分类。"""
import json
from pathlib import Path
from train.dataset.normalize import clean


def predict(record, text):
    """使用已登记模型对单条文本分类。

    Args:
        record (dict): 模型目录记录，须包含 ``id``、``name``、``path``、
            ``model`` 和 ``labels``。标签映射的键为字符串形式的类别编号。
        text (str): 待分类文本，推理前使用训练模块的规则清洗。

    Returns:
        dict: 包含 ``model_id``、``model_name``、``label``、``class_name``
        和 ``probabilities`` 的预测结果。概率列表包含所有登记标签的编号、
        名称和模型输出分数；``label`` 为字符串。分数不代表经过校准的置信度。

    Raises:
        ValueError: 模型类型不支持、缺少标签映射或预测标签未登记。
        OSError: 模型文件不存在或无法读取。

    Note:
        每次调用均从磁盘加载模型。RF 使用配套 TF-IDF 向量器，fastText
        沿用训练时的分词配置；BERT 在 CPU 上执行并关闭梯度计算，
        兼容自定义分类器及 Hugging Face 格式，仅读取本地文件。
    """
    path = Path(record["path"])
    text = clean(text)
    kind = record["model"]
    labels = record["labels"]
    if not labels:
        raise ValueError("模型缺少标签映射，请重新登记模型")
    if kind == "rf":
        import jieba
        import joblib
        model = joblib.load(path / "random_forest.joblib")
        vectorizer = joblib.load(path / "tfidf_vectorizer.joblib")
        probabilities = model.predict_proba(vectorizer.transform([" ".join(jieba.lcut(text))]))[0]
        scores = {str(int(key)): float(value) for key, value in zip(model.classes_, probabilities)}
        label = max(scores, key=scores.get)
    elif kind == "fasttext":
        import fasttext
        from train.dataset.exporters.fasttext import _char_text, _word_text
        config = json.loads((path / "training_config.json").read_text(encoding="utf-8"))
        transform = {"char": _char_text, "word": _word_text}[config["tokenizer"]]
        names, probabilities = fasttext.load_model(str(path / "fasttext.bin")).predict(transform(text), k=-1)
        by_name = {name.removeprefix("__label__"): float(value) for name, value in zip(names, probabilities)}
        scores = {key: by_name.get(name, 0.0) for key, name in labels.items()}
        label = max(scores, key=scores.get)
    elif kind == "bert":
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        weights = path / "final" if (path / "final").is_dir() else path
        tokenizer = AutoTokenizer.from_pretrained(str(weights), local_files_only=True)
        custom = (weights / "classifier.pt").exists()
        max_length = 512
        if custom:
            from transformers import BertConfig, BertModel
            from train.my_model import BertClassifierModel
            metadata = json.loads((weights / "classifier.json").read_text(encoding="utf-8"))
            model = BertClassifierModel(BertModel(BertConfig.from_pretrained(str(weights))), len(metadata["class_names"]))
            model.load_state_dict(torch.load(weights / "classifier.pt", map_location="cpu", weights_only=True))
            max_length = metadata["max_length"]
        else:
            model = AutoModelForSequenceClassification.from_pretrained(str(weights), local_files_only=True)
        model.eval()
        inputs = tokenizer([text], truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        with torch.inference_mode():
            logits = model(inputs["input_ids"], inputs["attention_mask"]) if custom else model(**inputs).logits
        label = str(logits.argmax(dim=-1).item())
        scores = {str(index): float(value) for index, value in enumerate(logits.softmax(dim=-1)[0].tolist())}
    else:
        raise ValueError("不支持的模型类型")
    if label not in labels:
        raise ValueError("预测标签不在模型标签映射中")
    return {"model_id": record["id"], "model_name": record["name"], "label": label, "class_name": labels[label],
            "probabilities": [{"label": key, "class_name": name, "probability": scores.get(key, 0.0)}
                              for key, name in labels.items()]}
