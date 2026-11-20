"""训练模块的输入/输出契约。"""

from ..logging import log_call as _log_call

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class TrainerError(Exception):
    """训练模块可预期的错误：数据缺失、参数非法、基座模型不存在。"""


@dataclass
class TrainerSpec:
    """一次训练的完整输入。

    :param kind: 训练类型，rf / fasttext / bert
    :param data_dir: dataset_tool 产出的目录（如 data/intent/bert）
    :param model_dir: 模型输出目录
    :param base_model: 基座模型路径或名称，bert 必填
    :param params: 各训练类型自己的超参
    """

    kind: str
    data_dir: Path
    model_dir: Path
    base_model: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    on_progress: Optional[Callable[[dict], None]] = None

    def __post_init__(self):
        """将数据和模型目录统一为 Path，并复制训练参数以隔离外部修改。

        Returns:
            None；原地规范化当前对象的字段。
        """
        with _log_call("train.trainer.spec.TrainerSpec.__post_init__", locals()):
            self.data_dir = Path(self.data_dir)
            self.model_dir = Path(self.model_dir)
            self.params = dict(self.params or {})


@dataclass
class TrainResult:
    kind: str
    metrics: Dict[str, Any]
    artifacts: Dict[str, int]
    duration_s: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """将训练结果转换成可序列化字典。

        Returns:
            包含 kind、metrics、artifacts、duration_s、warnings 的字典。
        """
        with _log_call("train.trainer.spec.TrainResult.to_dict", locals()):
            return {
                "kind": self.kind,
                "metrics": self.metrics,
                "artifacts": self.artifacts,
                "duration_s": self.duration_s,
                "warnings": self.warnings,
            }
