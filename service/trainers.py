"""三种模型的训练器：输入 dataset_tool 产出的目录，输出模型与指标。"""

from pathlib import Path


def _metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


def _read_tsv(path: Path):
    """读 `text<TAB>label` 的 txt，返回 (texts, labels)。"""
    texts, labels = [], []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue
            text, sep, label = line.rpartition("\t")
            if not sep:
                continue
            texts.append(text)
            labels.append(int(label))
    return texts, labels


def train_rf(prepared_dir: Path, model_dir: Path, params: dict, log):
    """随机森林：直接吃 TF-IDF 稀疏矩阵。"""
    import joblib
    import numpy as np
    import scipy.sparse as sp
    from sklearn.ensemble import RandomForestClassifier

    model_dir.mkdir(parents=True, exist_ok=True)
    load = lambda name: (sp.load_npz(prepared_dir / f"{name}_tfidf.npz"),
                         np.load(prepared_dir / f"{name}_labels.npy"))
    x_train, y_train = load("train")
    x_test, y_test = load("test")
    log(f"RF 输入 shape={x_train.shape}")

    clf = RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 100)),
        n_jobs=int(params.get("n_jobs", 1)),
        random_state=int(params.get("random_state", 42)),
    ).fit(x_train, y_train)
    joblib.dump(clf, model_dir / "random_forest.joblib")

    result = {"test": _metrics(y_test, clf.predict(x_test))}
    try:
        x_dev, y_dev = load("dev")
        result["dev"] = _metrics(y_dev, clf.predict(x_dev))
    except FileNotFoundError:
        pass
    return result


def train_fasttext(prepared_dir: Path, model_dir: Path, params: dict, log):
    """fastText：字符级或词级，二选一由 params.tokenizer 决定。"""
    import fasttext

    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = params.get("tokenizer", "char")
    if tokenizer not in ("char", "word"):
        raise ValueError(f"params.tokenizer 只支持 char/word，收到 {tokenizer}")

    train_path = prepared_dir / f"train_fastText_{tokenizer}.txt"
    log(f"fastText 训练文件: {train_path.name}")
    model = fasttext.train_supervised(
        input=str(train_path),
        epoch=int(params.get("epochs", 25)),
        lr=float(params.get("lr", 0.5)),
        dim=int(params.get("dim", 100)),
        wordNgrams=int(params.get("word_ngrams", 2)),
        # 默认 bucket=2000000 会让小数据集训出几百 MB 的模型，显式压小
        bucket=int(params.get("bucket", 50000)),
        minn=int(params.get("minn", 0)),
        maxn=int(params.get("maxn", 0)),
        verbose=0,
    )
    model.save_model(str(model_dir / "fasttext.bin"))

    result = {}
    for split in ("dev", "test"):
        path = prepared_dir / f"{split}_fastText_{tokenizer}.txt"
        if not path.is_file():
            continue
        y_true, y_pred = [], []
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                continue
            label, _, text = line.partition(" ")
            y_true.append(label)
            y_pred.append(model.predict(text, k=1)[0][0])
        result[split] = _metrics(y_true, y_pred)
    return result


def train_bert(prepared_dir: Path, model_dir: Path, params: dict, log):
    """BERT 微调：吃 dataset_tool 产出的 TSV。首次运行需要下载基座模型。"""
    import numpy as np
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments,
    )

    base_model = params.get("base_model", "google-bert/bert-base-chinese")
    max_length = int(params.get("max_length", 64))
    model_dir.mkdir(parents=True, exist_ok=True)

    log(f"加载基座模型: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def build(split):
        texts, labels = _read_tsv(prepared_dir / f"{split}.txt")
        ds = Dataset.from_dict({"text": texts, "label": labels})
        return ds.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
            batched=True,
        )

    train_set, dev_set, test_set = build("train"), build("dev"), build("test")
    num_labels = len(set(train_set["label"]))
    log(f"BERT 类别数={num_labels}  样本={len(train_set)}/{len(dev_set)}/{len(test_set)}")

    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels)
    arguments = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=float(params.get("lr", 3e-5)),
        per_device_train_batch_size=int(params.get("batch_size", 16)),
        per_device_eval_batch_size=int(params.get("eval_batch_size", 64)),
        num_train_epochs=float(params.get("epochs", 3)),
        logging_steps=int(params.get("logging_steps", 50)),
        report_to=[],
        disable_tqdm=True,
    )

    def compute(prediction):
        predicted = np.argmax(prediction.predictions, axis=-1)
        return _metrics(prediction.label_ids, predicted)

    trainer = Trainer(model=model, args=arguments, train_dataset=train_set,
                      eval_dataset=dev_set, compute_metrics=compute)
    trainer.train()

    result = {}
    for split, dataset in (("dev", dev_set), ("test", test_set)):
        scores = trainer.evaluate(dataset, metric_key_prefix=split)
        result[split] = {key.split("_", 1)[1]: round(float(value), 4)
                         for key, value in scores.items() if key.startswith(split + "_")}

    model.save_pretrained(str(model_dir / "final"))
    tokenizer.save_pretrained(str(model_dir / "final"))
    return result


TRAINERS = {"rf": train_rf, "fasttext": train_fasttext, "bert": train_bert}
