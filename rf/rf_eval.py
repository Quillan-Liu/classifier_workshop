"""加载训练好的模型，用独立 dev 集验证。"""

import json

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier

from . import rf_config as cfg
from .rf_train import scores


def main():
    """计算并保存 dev 集的准确率、精确率、召回率和 F1。"""
    x = load_npz(cfg.CSR_PATHS["dev"])
    y = np.load(cfg.LABEL_PATHS["dev"])
    model = joblib.load(cfg.MODEL_PATH)
    if not isinstance(model, RandomForestClassifier):
        raise ValueError("当前是旧模型，请先运行 rf_train.py 重新训练")
    if x.shape[0] != len(y) or x.shape[1] != model.n_features_in_:
        raise ValueError("dev 特征维度或标签数量不匹配")
    result = scores(y, model.predict(x))
    cfg.EVAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.EVAL_METRICS_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding=cfg.ENCODING
    )
    for name, value in result.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
