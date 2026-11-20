"""训练相关配置：基座模型路径、各模型训练产物的存储路径。

所有模型（含基座模型）都放在项目根目录的 models/ 下，换机器只改这一个文件。

    models/
    ├── base_models/bert-base-chinese/   基座模型
    ├── rf/{模型名}/                     随机森林产物
    ├── fasttext/{模型名}/               fastText 产物
    └── bert/{模型名}/                   BERT 微调产物
"""

from .logging import log_call as _log_call

from pathlib import Path

from .configs import defaults as _config_defaults

PROJECT_DIR = Path(__file__).resolve().parents[1]
ENCODING = "utf-8"

# ---------- 模型根目录 ----------

MODELS_DIR = PROJECT_DIR / "models"

# ---------- 基座模型 ----------

BASE_MODELS_DIR = MODELS_DIR / "base_models"

# bert 的基座模型目录（需包含 config.json / pytorch_model.bin / vocab.txt）
BERT_BASE_MODEL_DIR = BASE_MODELS_DIR / "bert-base-chinese"

# ---------- 训练产物 ----------

# 每类模型一个固定子目录，训练产物按模型名再分一层
MODEL_DIRS = {
    "rf": MODELS_DIR / "rf",
    "fasttext": MODELS_DIR / "fasttext",
    "bert": MODELS_DIR / "bert",
}

# ---------- 数据集处理产物 ----------

# dataset_tool 的默认输出根目录（未显式指定 --out 时）
PREPARED_DATA_DIR = PROJECT_DIR / "data" / "prepared"


def model_dir(model: str) -> Path:
    """查询指定模型类型的产物根目录，不创建目录。

    Args:
        model: 模型类型标识，如 rf、fasttext 或 bert。

    Returns:
        目录 Path；未知类型抛出 KeyError。

    Raises:
        KeyError: 模型类型未配置产物目录。
    """
    with _log_call("train.config.model_dir", locals()):
        if model not in MODEL_DIRS:
            raise KeyError(f"未知模型类型: {model}，可选 {list(MODEL_DIRS)}")
        return MODEL_DIRS[model]


def base_model(model: str, override: str | None = None) -> str | None:
    """解析 BERT 基座模型位置，优先使用调用方覆盖值。

    Args:
        model: 模型类型标识，如 rf、fasttext 或 bert。
        override: 调用方指定的基座模型路径或标识，优先于默认配置。

    Returns:
        BERT 的路径或标识字符串；其他模型返回 None。
    """
    with _log_call("train.config.base_model", locals()):
        if model != "bert":
            return None
        return override or str(BERT_BASE_MODEL_DIR)


# ---------- 训练默认参数 ----------

# 各模型的默认参数放在 configs/ 下，由各自的配置类定义：
#   configs/rf.py       RFConfig
#   configs/fasttext.py FastTextConfig
#   configs/bert.py     BertConfig


def defaults_for(model: str) -> dict:
    """获取指定模型类型的默认训练参数。

    Args:
        model: 模型类型标识，如 rf、fasttext 或 bert。

    Returns:
        默认参数字典。
    """
    with _log_call("train.config.defaults_for", locals()):
        return _config_defaults(model)


def merge_params(model: str, params: dict | None = None) -> dict:
    """将调用方训练参数覆盖到该模型的默认参数上。

    Args:
        model: 模型类型标识，如 rf、fasttext 或 bert。
        params: 训练参数字典；缺省字段使用模型默认配置。

    Returns:
        合并后的新字典，不修改传入参数。
    """
    with _log_call("train.config.merge_params", locals()):
        merged = defaults_for(model)
        merged.update({key: value for key, value in (params or {}).items() if value is not None})
        return merged


# ---------- 评估 ----------

EVAL_BATCH_SIZE = 64          # bert 推理时的批大小
EVAL_MAX_LENGTH = 64          # bert 推理时的截断长度，与训练默认值保持一致
EVAL_MAX_EXAMPLES = 20        # 结果里保留多少条错误样本
EVAL_METRICS_FILENAME = "eval_metrics.json"


# ---------- 服务 ----------

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8008
OPENAPI_PATH = PROJECT_DIR / "openapi.json"


if __name__ == "__main__":
    print("模型根目录 ->", MODELS_DIR)
    for kind in MODEL_DIRS:
        print(f"{kind:9} -> {model_dir(kind)}   默认={defaults_for(kind)}")
    print("基座模型   ->", base_model("bert"), "| 存在:", BERT_BASE_MODEL_DIR.exists())
