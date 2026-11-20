"""训练类型注册表与统一入口。"""

from ..logging import log_call as _log_call

from .bert import BertTrainer
from .fasttext import FastTextTrainer
from .rf import RandomForestTrainer
from .spec import TrainResult, TrainerError, TrainerSpec

_TRAINERS = {
    "rf": RandomForestTrainer,
    "fasttext": FastTextTrainer,
    "bert": BertTrainer,
}


def supported() -> list[str]:
    """列出已注册的训练器类型。

    Returns:
        模型类型字符串列表。
    """
    with _log_call("train.trainer.registry.supported", locals()):
        return list(_TRAINERS)


def create_trainer(spec: TrainerSpec):
    """根据训练输入的 kind 选择并初始化训练器。

    Args:
        spec: 训练输入契约，包含类型、数据目录、模型目录、基座及超参数。

    Returns:
        对应的训练器实例；未知类型抛出 TrainerError。

    Raises:
        TrainerError: 训练器类型未注册。
    """
    with _log_call("train.trainer.registry.create_trainer", locals()):
        trainer_cls = _TRAINERS.get(spec.kind)
        if trainer_cls is None:
            raise TrainerError(f"不支持的训练类型 {spec.kind!r}，可选 {list(_TRAINERS)}")
        return trainer_cls(spec)


def train(spec: TrainerSpec) -> TrainResult:
    """通过类型注册表创建训练器并执行完整训练流程。

    Args:
        spec: 训练输入契约，包含类型、数据目录、模型目录、基座及超参数。

    Returns:
        包含指标、产物、耗时及警告的 TrainResult。
    """
    with _log_call("train.trainer.registry.train", locals()):
        return create_trainer(spec).run()
