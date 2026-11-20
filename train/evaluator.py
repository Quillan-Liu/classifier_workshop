"""验证模块：输入模型目录 + 测试集产物，输出测试结果。

沿用项目里 `rf/rf_eval.py`、`ft/infer.py`、`bert/src/bert_infer.py` 的约定：
指标口径是 accuracy / precision / recall / f1（macro，zero_division=0）。

用法：
    python -m train.evaluator --model models/rf/rf_20261119_201758_intent_764ba0 --test data/intent/rf
    python -m train.evaluator --model models/fasttext/ft_xxx --test data/intent/fasttext
    python -m train.evaluator --model models/bert/bert_xxx --test data/intent/bert
"""

from .logging import log_call as _log_call

import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_fscore_support,
)

from .config import (
    ENCODING, EVAL_BATCH_SIZE, EVAL_MAX_EXAMPLES, EVAL_MAX_LENGTH, EVAL_METRICS_FILENAME,
)
from .logging import get_logger

logger = get_logger(__name__)

# 各类型模型的特征文件名
RF_FILES = ("random_forest.joblib",)
FASTTEXT_FILES = ("fasttext.bin",)
BERT_FILES = ("config.json", "model.safetensors")


# ---------------- 基础工具 ----------------


def scores(y_true, y_pred) -> dict:
    """计算准确率、宏平均精确率/召回率/F1 和加权 F1，结果保留四位小数。

    Args:
        y_true: 真实标签序列。
        y_pred: 模型预测标签序列，与真实标签一一对应。

    Returns:
        包含 accuracy、precision、recall、f1、f1_weighted 的字典。
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted",
                                             zero_division=0)), 4),
    }


def detect_kind(model_path: Path) -> str:
    """根据模型产物文件识别类型，无法匹配时尝试解析模型名。

    Args:
        model_path: 模型产物目录路径。

    Returns:
        rf、fasttext 或 bert；无法识别时抛出 ValueError。

    Raises:
        ValueError: 产物文件和名称均无法确定模型类型。
    """
    with _log_call("train.evaluator.detect_kind", locals()):
        model_path = Path(model_path)
        if (model_path / "final" / "classifier.pt").exists() or (model_path / "classifier.pt").exists():
            return "bert"
        if any((model_path / name).exists() for name in RF_FILES):
            return "rf"
        if any((model_path / name).exists() for name in FASTTEXT_FILES):
            return "fasttext"
        if (model_path / "config.json").exists() and (
                (model_path / "model.safetensors").exists()
                or (model_path / "pytorch_model.bin").exists()):
            return "bert"
        from .naming import parse_model_name

        try:
            return parse_model_name(model_path.name).model
        except ValueError:
            raise ValueError(
                f"无法判断模型类型: {model_path}（目录里没有 random_forest.joblib / "
                f"fasttext.bin / config.json）"
            )


def _read_tsv(path: Path):
    """读取以最后一个制表符分隔文本和整数标签的数据，跳过空行及无分隔符行。

    Args:
        path: 待读取的数据文件、目录或配置文件路径。

    Returns:
        二元组 (文本列表, 整数标签列表)。
    """
    with _log_call("train.evaluator._read_tsv", locals()):
        texts, labels = [], []
        with open(path, encoding=ENCODING) as file:
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


def _class_names(test_dir: Path, num_classes: int) -> dict:
    """读取测试目录或上级目录的 class.txt；类别数不符时使用编号作为名称。

    Args:
        test_dir: 数据处理后测试文件所在目录。
        num_classes: 期望的类别数量。

    Returns:
        {类别编号: 类别名称} 字典。
    """
    with _log_call("train.evaluator._class_names", locals()):
        for candidate in (Path(test_dir) / "class.txt", Path(test_dir).parent / "class.txt"):
            if candidate.is_file():
                names = [line.strip() for line in
                         candidate.read_text(encoding=ENCODING).splitlines() if line.strip()]
                if len(names) == num_classes:
                    return {index: name for index, name in enumerate(names)}
        return {index: str(index) for index in range(num_classes)}


# ---------------- 三种模型的预测 ----------------


def _predict_rf(model_path: Path, test_dir: Path):
    """加载随机森林和测试集 TF-IDF 特征，校验行数及特征维度后预测。

    Args:
        model_path: 模型产物目录路径。
        test_dir: 数据处理后测试文件所在目录。

    Returns:
        三元组 (预测标签, 真实标签, None)；维度不匹配时抛出 ValueError。

    Raises:
        ValueError: 特征行数与标签数量不符，或特征维度与模型不符。
    """
    with _log_call("train.evaluator._predict_rf", locals()):
        import joblib
        import numpy as np
        from scipy.sparse import load_npz

        model = joblib.load(model_path / "random_forest.joblib")
        x = load_npz(test_dir / "test_tfidf.npz")
        y_true = np.load(test_dir / "test_labels.npy")
        if x.shape[0] != len(y_true):
            raise ValueError(f"特征与标签数量不一致: {x.shape[0]} vs {len(y_true)}")
        if x.shape[1] != getattr(model, "n_features_in_", x.shape[1]):
            raise ValueError(
                f"特征维度不匹配: 模型 {model.n_features_in_} vs 测试集 {x.shape[1]}，"
                f"测试集可能来自另一次数据集处理"
            )
        return model.predict(x), y_true, None


def _predict_fasttext(model_path: Path, test_dir: Path):
    """按模型保存的分词方式选择 fastText 测试文件并逐条预测。

    Args:
        model_path: 模型产物目录路径。
        test_dir: 数据处理后测试文件所在目录。

    Returns:
        三元组 (预测类名列表, 真实类名列表, 文本列表)。

    Raises:
        FileNotFoundError: 未找到 fastText 测试文件。
    """
    with _log_call("train.evaluator._predict_fasttext", locals()):
        import fasttext

        candidates = sorted(test_dir.glob("test_fastText_*.txt"))
        if not candidates:
            raise FileNotFoundError(f"{test_dir} 下没有 test_fastText_*.txt")
        config_path = model_path / "training_config.json"
        if config_path.is_file():
            tokenizer = json.loads(config_path.read_text(encoding="utf-8"))["tokenizer"]
            if tokenizer not in ("char", "word"):
                raise ValueError(f"无效的 fastText 分词方式: {tokenizer!r}")
            test_file = test_dir / f"test_fastText_{tokenizer}.txt"
            if not test_file.is_file():
                raise FileNotFoundError(test_file)
        elif len(candidates) == 1:
            test_file = candidates[0]
        else:
            raise ValueError("模型缺少 training_config.json，无法确定使用 char 还是 word 测试文件")
        logger.info("fastText 测试文件=%s", test_file.name)

        model = fasttext.load_model(str(model_path / "fasttext.bin"))
        texts, labels, predictions = [], [], []
        for line in open(test_file, encoding=ENCODING):
            line = line.rstrip("\n")
            if not line:
                continue
            label, _, text = line.partition(" ")
            texts.append(text)
            labels.append(label.removeprefix("__label__"))
            predictions.append(model.predict(text, k=1)[0][0].removeprefix("__label__"))
        return predictions, labels, texts


def _predict_bert(model_path: Path, test_dir: Path):
    """加载自定义分类器或 Hugging Face BERT，按批次预测测试文本。

    Args:
        model_path: 模型产物目录路径。
        test_dir: 数据处理后测试文件所在目录。

    Returns:
        三元组 (预测编号列表, 真实编号列表, 文本列表)。
    """
    with _log_call("train.evaluator._predict_bert", locals()):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        weights_dir = model_path / "final" if (model_path / "final").exists() else model_path
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(str(weights_dir))
        custom = (weights_dir / "classifier.pt").exists()
        max_length = EVAL_MAX_LENGTH
        if custom:
            from transformers import BertConfig, BertModel
            from bert.src.bert_classifier_model import BertClassifierModel

            metadata = json.loads((weights_dir / "classifier.json").read_text(encoding="utf-8"))
            model = BertClassifierModel(BertModel(BertConfig.from_pretrained(str(weights_dir))),
                                        len(metadata["class_names"])).to(device)
            model.load_state_dict(torch.load(weights_dir / "classifier.pt", map_location=device, weights_only=True))
            max_length = metadata["max_length"]
        else:
            model = AutoModelForSequenceClassification.from_pretrained(str(weights_dir)).to(device)
        model.eval()

        texts, labels = _read_tsv(test_dir / "test.txt")
        batch_size = EVAL_BATCH_SIZE
        predictions = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            inputs = tokenizer(chunk, truncation=True, padding=True,
                               max_length=max_length, return_tensors="pt").to(device)
            with torch.inference_mode():
                logits = (model(inputs["input_ids"], inputs["attention_mask"]) if custom
                          else model(**inputs).logits)
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        return predictions, labels, texts


_PREDICTORS = {"rf": _predict_rf, "fasttext": _predict_fasttext, "bert": _predict_bert}


# ---------------- 对外入口 ----------------


def evaluate(model_path, test_dir, save: bool = True,
             max_examples: int = EVAL_MAX_EXAMPLES) -> dict:
    """识别并加载模型，运行测试集，生成指标、分类报告、混淆矩阵和错误示例。

    Args:
        model_path: 模型产物目录路径。
        test_dir: 数据处理后测试文件所在目录。
        save: 是否将完整评估结果写入模型目录。
        max_examples: 报告中最多保留的错误文本示例数。

    Returns:
        完整评估字典；save=true 时还写入模型目录的评估 JSON 文件。

    Raises:
        FileNotFoundError: 模型目录或测试目录不存在。
        ValueError: 模型类型、预测数量或测试数据无法满足评估要求。
    """
    with _log_call("train.evaluator.evaluate", locals()):
        model_path, test_dir = Path(model_path), Path(test_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"模型目录不存在: {model_path}")
        if not test_dir.exists():
            raise FileNotFoundError(f"测试集目录不存在: {test_dir}")

        kind = detect_kind(model_path)
        logger.info("开始评估 model=%s kind=%s test=%s", model_path.name, kind, test_dir)

        y_pred, y_true, texts = _PREDICTORS[kind](model_path, test_dir)
        if len(y_pred) != len(y_true):
            raise ValueError(f"预测数与标签数不一致: {len(y_pred)} vs {len(y_true)}")

        # fastText 的标签是类名字符串，统一映射成编号，便于和 class.txt 对齐
        if kind == "fasttext":
            names = sorted(set(y_true) | set(y_pred))
            mapping = {name: index for index, name in enumerate(names)}
            y_true = [mapping[item] for item in y_true]
            y_pred = [mapping[item] for item in y_pred]
            class_names = {index: name for index, name in enumerate(names)}
        else:
            class_names = _class_names(test_dir, max(max(y_true), max(y_pred)) + 1)

        result = {
            "model": model_path.name,
            "kind": kind,
            "model_path": str(model_path),
            "test_dir": str(test_dir),
            "rows": len(y_true),
            "classes": class_names,
            "scores": scores(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "report": classification_report(
                y_true, y_pred, target_names=[class_names[i] for i in sorted(class_names)],
                zero_division=0, output_dict=True, digits=4),
        }

        wrong = [{"text": text, "true": class_names[true], "pred": class_names[pred]}
                 for text, true, pred in zip(texts or [], y_true, y_pred) if true != pred]
        result["wrong_examples"] = wrong[:max_examples]
        result["wrong_count"] = len(wrong)

        if save:
            target = model_path / EVAL_METRICS_FILENAME
            target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding=ENCODING)
            logger.info("评估结果已保存 %s", target)

        logger.info("评估完成 model=%s rows=%s scores=%s",
                    model_path.name, result["rows"], result["scores"])
        return result


def print_report(result: dict) -> None:
    """将评估指标、分类报告、混淆矩阵及部分错误示例打印到终端。

    Args:
        result: 包含指标、分类报告、混淆矩阵及错误示例的评估结果。

    Returns:
        None。
    """
    with _log_call("train.evaluator.print_report", locals()):
        classes = {int(index): name for index, name in result["classes"].items()}
        names = [classes[index] for index in sorted(classes)]
        print(f"模型: {result['model']}  类型: {result['kind']}  测试样本: {result['rows']}")
        for name, value in result["scores"].items():
            print(f"{name}: {value:.4f}")
        print(f"\n分类报告:")
        print(classification_report_from_dict(result["report"], names))
        print("混淆矩阵（行=真实，列=预测）:")
        header = "      " + " ".join(f"{name[:6]:>7}" for name in names)
        print(header)
        for name, row in zip(names, result["confusion_matrix"]):
            print(f"{name[:6]:>6}" + " ".join(f"{value:>7}" for value in row))
        if result["wrong_count"]:
            print(f"\n错误样本 {result['wrong_count']} 条，示例:")
            for item in result["wrong_examples"][:5]:
                print(f"  [{item['true']} -> {item['pred']}] {item['text']}")


def classification_report_from_dict(report: dict, classes: list) -> str:
    """将 sklearn 分类报告字典排版成文本表格。

    Args:
        report: sklearn output_dict 形式的分类报告。
        classes: 决定报告显示顺序的类别名称列表。

    Returns:
        按给定类别顺序排列的报告字符串。
    """
    with _log_call("train.evaluator.classification_report_from_dict", locals()):
        lines = [f"{'class':<16}{'precision':>10}{'recall':>8}{'f1':>8}{'support':>9}"]
        for name in classes:
            row = report.get(name, {})
            lines.append(f"{name:<16}{row.get('precision', 0):>10.4f}"
                         f"{row.get('recall', 0):>8.4f}{row.get('f1-score', 0):>8.4f}"
                         f"{int(row.get('support', 0)):>9}")
        for key in ("macro avg", "weighted avg"):
            row = report.get(key, {})
            lines.append(f"{key:<16}{row.get('precision', 0):>10.4f}"
                         f"{row.get('recall', 0):>8.4f}{row.get('f1-score', 0):>8.4f}"
                         f"{int(row.get('support', 0)):>9}")
        return "\n".join(lines)


def main(argv=None):
    """解析评估命令行参数，执行测试并打印报告。

    Args:
        argv: 命令行参数列表；None 表示读取当前进程参数。

    Returns:
        完整评估结果字典。
    """
    with _log_call("train.evaluator.main", locals()):
        import argparse

        parser = argparse.ArgumentParser(prog="train.evaluator", description="用测试集验证模型")
        parser.add_argument("--model", required=True, help="模型目录，如 models/rf/rf_xxx")
        parser.add_argument("--test", required=True, help="测试集产物目录，如 data/intent/rf")
        parser.add_argument("--no-save", action="store_true", help="不写 eval_metrics.json")
        parser.add_argument("--examples", type=int, default=EVAL_MAX_EXAMPLES,
                            help="保留多少条错误样本")
        args = parser.parse_args(argv)

        result = evaluate(args.model, args.test, save=not args.no_save,
                          max_examples=args.examples)
        print_report(result)
        return result


if __name__ == "__main__":
    main()
