"""三类模型的训练配置，各自命名、各自被对应 trainer 调用。"""


from enum import Enum

from .bert import BertConfig
from .fasttext import FastTextConfig
from .rf import RFConfig

CONFIGS = {
    "rf": RFConfig,
    "fasttext": FastTextConfig,
    "bert": BertConfig,
}


class ModelKind(str, Enum):
    """训练类型只有这三种；接口层用它生成下拉选项（OpenAPI enum）。"""

    rf = "rf"
    fasttext = "fasttext"
    bert = "bert"


# 给前端渲染选项表用：value 是提交值，label 是展示名，description 是说明
MODEL_OPTIONS = [
    {"value": "rf", "label": "随机森林", "description": "TF-IDF + RandomForest，秒级训练"},
    {"value": "fasttext", "label": "fastText", "description": "字符级/词级 fastText，秒级训练"},
    {"value": "bert", "label": "BERT 微调", "description": "bert-base 微调，需要 GPU，分钟级"},
]


def get_config(model: str, params: dict | None = None):
    """按模型类型构建配置，使用传入参数覆盖已知字段的默认值。

    Args:
        model: 模型类型标识，如 rf、fasttext 或 bert。
        params: 训练参数字典；缺省字段使用模型默认配置。

    Returns:
        对应配置类实例；未知类型抛出 ValueError。

    Raises:
        ValueError: 模型类型未注册配置类。
    """
    config_cls = CONFIGS.get(model)
    if config_cls is None:
        raise ValueError(f"不支持的模型类型: {model!r}，可选 {list(CONFIGS)}")
    return config_cls.from_params(params)


def defaults(model: str) -> dict:
    """查询指定模型类型的默认配置并转为字典。

    Args:
        model: 模型类型标识，如 rf、fasttext 或 bert。

    Returns:
        默认训练参数字典。
    """
    return get_config(model).to_dict()


if __name__ == "__main__":
    for name in CONFIGS:
        print(f"{name:9}", defaults(name))
