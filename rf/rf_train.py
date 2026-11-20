"""训练随机森林：训练集内部划分验证集，不读取 dev/test。"""

import json

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from . import rf_config as cfg


def scores(y_true, y_pred):
    """返回准确率、宏平均精确率、召回率和 F1。"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision, "recall": recall, "f1": f1}


def train(x_train, y_train, x_val, y_val):
    """分批增加树，每轮验证，并保留验证准确率最高的轮次。"""
    if cfg.RF_N_ESTIMATORS % cfg.RF_TREES_PER_STEP:
        raise ValueError("每轮树数必须能被每批树数整除")
    model = RandomForestClassifier(warm_start=True, n_jobs=cfg.RF_N_JOBS,
                                   random_state=cfg.RF_RANDOM_STATE)
    history, best_accuracy, best_trees = [], -1, 0
    total = cfg.RF_MAX_ROUNDS * cfg.RF_N_ESTIMATORS
    with tqdm(total=total, desc="RF 训练", unit="棵") as bar:
        for trees in range(cfg.RF_TREES_PER_STEP, total + 1, cfg.RF_TREES_PER_STEP):
            model.n_estimators = trees
            model.fit(x_train, y_train)
            bar.update(cfg.RF_TREES_PER_STEP)
            if trees % cfg.RF_N_ESTIMATORS:
                continue
            result = scores(y_val, model.predict(x_val))
            history.append({"round": len(history) + 1, "trees": trees, **result})
            bar.write(f"第 {len(history)} 轮：{result}")
            if result["accuracy"] > best_accuracy:
                best_accuracy, best_trees = result["accuracy"], trees
            if result["accuracy"] >= cfg.RF_EARLY_STOP_ACCURACY:
                break
    model.estimators_ = model.estimators_[:best_trees]
    model.n_estimators = best_trees
    return model, history


def main():
    """读取训练 CSR，划分内部验证集，训练并保存模型。"""
    x = load_npz(cfg.CSR_PATHS["train"])
    y = np.load(cfg.LABEL_PATHS["train"])
    if x.shape[0] != len(y):
        raise ValueError("训练特征与标签数量不一致")
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=cfg.TRAIN_VALIDATION_SIZE,
        stratify=y, random_state=cfg.RF_RANDOM_STATE,
    )
    model, history = train(x_train, y_train, x_val, y_val)
    result = {"train_samples": len(y_train), "val_samples": len(y_val),
              "history": history, "validation": scores(y_val, model.predict(x_val))}
    cfg.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.TRAIN_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg.MODEL_PATH)
    cfg.TRAIN_METRICS_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding=cfg.ENCODING
    )
    print(f"模型：{cfg.MODEL_PATH}\n内部验证指标：{cfg.TRAIN_METRICS_PATH}")


if __name__ == "__main__":
    main()
