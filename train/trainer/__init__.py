"""训练模块：输入基座模型路径 + 数据集产物 + 训练类型，输出模型与指标。

    from .trainer import TrainerSpec, train

    result = train(TrainerSpec(
        kind="bert",
        data_dir="D:/tmf/data/intent/bert",
        model_dir="D:/tmf/runs/demo/model",
        base_model="D:/models/bert-base-chinese",
        params={"epochs": 3},
    ))
    print(result.metrics)
"""

from .registry import create_trainer, supported, train
from .spec import TrainResult, TrainerError, TrainerSpec

__all__ = ["TrainerSpec", "TrainResult", "TrainerError", "train", "create_trainer", "supported"]
