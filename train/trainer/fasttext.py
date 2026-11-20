"""fastText：字符级或词级，由 FastTextConfig.tokenizer 决定。"""

from ..logging import log_call as _log_call, format_number

from typing import Tuple
import json
import re
import subprocess
import sys
from pathlib import Path

from ..configs import get_config
from .base import BaseTrainer


class FastTextTrainer(BaseTrainer):
    kind = "fasttext"

    def _config(self):
        """从训练输入参数构建 fastText 配置。

        Returns:
            FastTextConfig 实例。
        """
        with _log_call("train.trainer.fasttext.FastTextTrainer._config", locals()):
            return get_config("fasttext", self.spec.params)

    def required(self) -> Tuple[str, ...]:
        """根据字符级或词级分词配置确定训练输入文件。

        Returns:
            必需训练文件名元组。
        """
        with _log_call("train.trainer.fasttext.FastTextTrainer.required", locals()):
            tokenizer = self._config().tokenizer
            return (f"train_fastText_{tokenizer}.txt",)

    def _metrics(self, y_true, y_pred):
        """计算准确率和宏平均 F1，保留四位小数。

        Args:
            y_true: 真实标签序列。
            y_pred: 模型预测标签序列，与真实标签一一对应。

        Returns:
            包含 accuracy 和 macro_f1 的字典。
        """
        from ..evaluator import scores
        return scores(y_true, y_pred)

    def fit(self) -> dict:
        """训练并保存 fasttext.bin，对存在的 dev/test 分组计算指标。

        Returns:
            以分组名为键的指标字典；缺少评估文件时记录警告并跳过。
        """
        with _log_call("train.trainer.fasttext.FastTextTrainer.fit", locals()):
            import fasttext

            data_dir = self.spec.data_dir
            cfg = self._config()
            tokenizer = cfg.tokenizer
            train_path = data_dir / f"train_fastText_{tokenizer}.txt"
            self.logger.info("fastText 训练文件=%s 配置=%s", train_path, cfg.to_dict())
            self.warn("fastText 保留原生训练：实时记录原生进度，训练结束后评估 train/dev/test；不提供逐轮评估。")

            arguments = {"input": str(train_path.resolve()), "verbose": cfg.verbose, "thread": cfg.thread}
            parameter_names = {"epochs": "epoch", "lr": "lr", "dim": "dim",
                               "word_ngrams": "wordNgrams", "bucket": "bucket",
                               "minn": "minn", "maxn": "maxn"}
            if cfg.autotune:
                validation_path = data_dir / f"dev_fastText_{tokenizer}.txt"
                if not validation_path.is_file() or not validation_path.stat().st_size:
                    raise ValueError("自动调参选优需要非空的验证集 dev 文件")
                arguments.update(autotuneValidationFile=str(validation_path),
                                 autotuneDuration=cfg.autotune_duration,
                                 thread=cfg.thread, verbose=cfg.verbose,
                                 autotuneModelSize=cfg.autotuneModelSize)
                self.report(stage="autotune_started", duration_seconds=cfg.autotune_duration,
                            validation_file=validation_path.name)
            else:
                arguments.update({target: getattr(cfg, source) for source, target in parameter_names.items()})
            model_path = self.spec.model_dir / "fasttext.bin"
            self.report(stage="native_training_started")
            # 独立进程隔离 C++ stderr，避免并发任务重定向整个服务的输出。
            with subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("fasttext_worker.py")),
                 json.dumps(arguments), str(model_path.resolve())],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            ) as process:
                line, recent, last_percent = "", [], -1
                while True:
                    char = process.stdout.read(1)
                    if not char and not line:
                        break
                    if char and char not in "\r\n":
                        line += char
                        continue
                    if line.strip():
                        readable = re.sub(
                            r"(loss:\s*)?((?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
                            lambda match: (match[1] or "") + format_number(
                                float(match[2]), loss=bool(match[1])), line.strip())
                        readable = readable.replace("Progress:", "进度：").replace("words/sec/thread:", "每线程每秒词数：").replace("lr:", "学习率：").replace("loss:", "损失：").replace("eta:", "预计剩余时间：")
                        self.logger.info("fastText %s", readable)
                        recent = (recent + [line.strip()])[-20:]
                        match = re.search(r"Progress:\s*([\d.]+)%", line)
                        if match and int(float(match[1])) != last_percent:
                            last_percent = int(float(match[1]))
                            details = {}
                            for key, pattern in (("native_loss", r"loss:\s*([\d.eE+-]+)"), ("learning_rate", r"lr:\s*([\d.eE+-]+)")):
                                value = re.search(pattern, line)
                                if value:
                                    details[key] = float(value[1])
                            self.report(stage="native_training", progress_percent=float(match[1]), **details)
                    line = ""
                if process.wait():
                    raise RuntimeError("fastText 原生训练失败：\n" + "\n".join(recent))
            model = fasttext.load_model(str(model_path))
            effective = json.loads((self.spec.model_dir / "training_config.json").read_text(encoding="utf-8"))
            self.report(stage="parameters_selected", tokenizer=tokenizer, autotune=cfg.autotune,
                        parameters=effective)
            (self.spec.model_dir / "training_config.json").write_text(
                json.dumps({**cfg.to_dict(), **effective}, ensure_ascii=False, indent=2), encoding="utf-8")

            metrics = {}
            for split in ("train", "dev", "test"):
                path = data_dir / f"{split}_fastText_{tokenizer}.txt"
                if not path.is_file():
                    self.warn(f"缺少 {split} 文件，跳过该 split: {path}")
                    continue
                y_true, y_pred = [], []
                with path.open(encoding="utf-8") as source:
                    for line in source:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        label, _, text = line.partition(" ")
                        y_true.append(label)
                        y_pred.append(model.predict(text, k=1)[0][0])
                metrics[split] = self._metrics(y_true, y_pred)
            metrics["selected_epoch"] = int(effective["epochs"])
            self.report(stage="completed", **metrics)
            metrics["parameters"] = effective
            return metrics
