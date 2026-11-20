"""训练基类：校验数据 -> 训练 -> 收集产物，统一打日志和兜底异常。"""

from ..logging import log_call as _log_call

import json
import time
from abc import ABC, abstractmethod
from typing import Tuple

from ..logging import get_logger, training_log, training_log_dir, format_progress

from .spec import TrainResult, TrainerError


class BaseTrainer(ABC):
    kind: str = ""
    required_files: Tuple[str, ...] = ()

    def __init__(self, spec):
        """保存训练输入，初始化该类型的日志记录器和警告列表。

        Args:
            spec: 训练输入契约，包含类型、数据目录、模型目录、基座及超参数。

        Returns:
            None。
        """
        with _log_call("train.trainer.base.BaseTrainer.__init__", locals()):
            self.spec = spec
            self.logger = get_logger(f"trainer.{self.kind}")
            self.warnings: list[str] = []

    # ---------- 子类接口 ----------

    def required(self) -> Tuple[str, ...]:
        """声明当前训练器启动前必须存在的数据文件。

        Returns:
            相对于 data_dir 的文件名元组。
        """
        with _log_call("train.trainer.base.BaseTrainer.required", locals()):
            return self.required_files

    @abstractmethod
    def fit(self) -> dict:
        """定义由子类实现的训练过程及指标输出契约。

        Returns:
            训练指标字典；具体实现负责保存模型产物。
        """

    # ---------- 公共流程 ----------

    def report(self, **record):
        """保存每次训练进度，并同步到服务任务供前端轮询。"""
        self.logger.info("%s", format_progress(record))
        with (training_log_dir(self.spec.model_dir.name) / "training_history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.spec.on_progress:
            self.spec.on_progress(record)

    def warn(self, message: str) -> None:
        """保存警告消息并写入日志，供最终训练结果一并返回。

        Args:
            message: 要记录的日志或警告文本。

        Returns:
            None。
        """
        with _log_call("train.trainer.base.BaseTrainer.warn", locals()):
            self.warnings.append(message)
            self.logger.warning(message)

    def validate(self) -> None:
        """校验数据目录、必需文件和 BERT 基座设置，并创建模型输出目录。

        Returns:
            None；不满足训练条件时抛出 TrainerError。

        Raises:
            TrainerError: 数据目录、必需文件或 BERT 基座设置缺失。
        """
        with _log_call("train.trainer.base.BaseTrainer.validate", locals()):
            data_dir = self.spec.data_dir
            if not data_dir.exists():
                raise TrainerError(f"数据集产物目录不存在: {data_dir}")

            missing = [name for name in self.required() if not (data_dir / name).exists()]
            if missing:
                raise TrainerError(
                    f"{self.kind} 缺少数据文件 {missing}（目录 {data_dir}）。"
                    f"请先用 dataset_tool 处理数据集且 --use {self.kind}"
                )

            if self.kind == "bert" and not self.spec.base_model:
                raise TrainerError("bert 训练必须提供基座模型：本地目录路径或模型名")

            self.spec.model_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> TrainResult:
        """校验输入、执行子类训练、统计耗时并收集产物和警告。

        Returns:
            TrainResult；训练异常统一包装或保留为 TrainerError。

        Raises:
            TrainerError: 校验或训练失败，非业务训练异常包装为此类型。
        """
        with training_log(self.spec.model_dir.name), _log_call("train.trainer.base.BaseTrainer.run", locals()):
            self.validate()
            started = time.perf_counter()
            self.logger.info(
                "开始训练 kind=%s data=%s out=%s base=%s params=%s",
                self.kind, self.spec.data_dir, self.spec.model_dir,
                self.spec.base_model or "-", self.spec.params,
            )
            try:
                metrics = self.fit()
            except TrainerError:
                self.logger.error("训练中止 kind=%s 原因见上", self.kind)
                raise
            except Exception as exc:  # noqa: BLE001 - 统一包装成可读错误
                self.logger.exception("训练异常 kind=%s", self.kind)
                raise TrainerError(f"{self.kind} 训练失败: {type(exc).__name__}: {exc}") from exc

            (self.spec.model_dir / "training_summary.json").write_text(
                json.dumps({"selected_epoch": metrics["selected_epoch"]}, ensure_ascii=False, indent=2),
                encoding="utf-8")

            duration = time.perf_counter() - started
            artifacts = {
                str(path.relative_to(self.spec.model_dir)): path.stat().st_size
                for path in sorted(self.spec.model_dir.rglob("*")) if path.is_file()
            }
            self.logger.info("训练完成 · 模型类型=%s · 耗时=%.1f 秒\n%s",
                             self.kind, duration, format_progress({"stage": "completed", **metrics}))
            return TrainResult(
                kind=self.kind,
                metrics=metrics,
                artifacts=artifacts,
                duration_s=round(duration, 2),
                warnings=self.warnings,
            )
