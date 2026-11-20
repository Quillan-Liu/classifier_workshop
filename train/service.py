"""服务层：一次训练 = 处理数据集 -> 按模型类型路由 -> 训练模型 -> 测试集验证。

不依赖 FastAPI；任务保存在内存中，成功模型通过 ModelStore 持久化登记。
"""

from .logging import log_call as _log_call

import shutil
import threading
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import base_model as resolve_base_model, merge_params, model_dir as output_dir_for
from .dataset import prepare
from .evaluator import evaluate
from .logging import get_logger, training_log, format_progress
from .naming import new_model_name
from model_management import ModelStore
from model_management.purposes import purpose_code
from .trainer import TrainerSpec, create_trainer

logger = get_logger(__name__)
WORK_ROOT = Path(__file__).resolve().parents[1] / "runs"

STAGE_DATA = "处理数据集"
STAGE_ROUTE = "路由训练器"
STAGE_TRAIN = "训练模型"
STAGE_TEST = "测试模型"


def _now() -> str:
    """获取带本地时区的当前时间，用于任务时间戳和日志。

    Returns:
        精确到秒的 ISO 8601 字符串。
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")


class JobNotFound(Exception):
    """任务不存在。接口层据此返回 404。"""


class StageError(Exception):
    """某个阶段失败，带上阶段名便于定位。"""

    def __init__(self, stage: str, cause: Exception):
        """保存失败阶段和原始异常，构造可读的阶段错误消息。

        Args:
            stage: 发生异常的阶段名称。
            cause: 导致阶段失败的原始异常。

        Returns:
            None。
        """
        with _log_call("train.service.StageError.__init__", locals()):
            self.stage = stage
            self.cause = cause
            super().__init__(f"{stage}失败: {type(cause).__name__}: {cause}")


@dataclass
class Job:
    id: str
    model: str
    source: str
    test_source: str = ""
    name: str = ""
    purpose: str = "default"
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    metrics: Optional[dict] = None
    artifacts: Optional[dict] = None
    stages: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    log: List[str] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)
    progress: dict = field(default_factory=dict)

    def trace(self, message: str) -> None:
        """给当前任务追加一条带时间戳的日志。

        Args:
            message: 要记录的日志或警告文本。

        Returns:
            None。
        """
        self.log.append(f"[{_now()}] {message}")

    def to_dict(self, with_log: bool = False) -> dict:
        """将任务转换成接口可返回的字典，按需包含日志副本。

        Args:
            with_log: 是否在返回字典中包含日志列表副本。

        Returns:
            任务标识、用途、状态、指标、产物、阶段耗时等信息。
        """
        data = {
            "id": self.id, "name": self.name, "model": self.model, "source": self.source,
            "test_source": self.test_source,
            "purpose": self.purpose,
            "status": self.status, "created_at": self.created_at,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "metrics": self.metrics, "artifacts": self.artifacts,
            "stages": self.stages, "error": self.error,
            "history": list(self.history), "progress": dict(self.progress),
        }
        if with_log:
            data["log"] = list(self.log)
        return data


class TrainService:
    """提交训练、查询任务。"""

    def __init__(self, work_root: Path = WORK_ROOT, max_workers: int = 2, model_store=None):
        """初始化任务内存存储、线程锁、训练线程池和模型登记存储。

        Args:
            work_root: 训练任务中间产物的工作根目录。
            max_workers: 训练线程池可同时执行的最大任务数。
            model_store: 模型登记存储实例；未提供时使用默认 SQLite 存储。

        Returns:
            None。
        """
        with _log_call("train.service.TrainService.__init__", locals()):
            self.work_root = Path(work_root)
            self.model_store = model_store if model_store is not None else ModelStore()
            self._jobs: Dict[str, Job] = {}
            self._lock = threading.RLock()
            self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="train")

    # ---------- 对外用例 ----------

    def submit(self, model: str, train_path: str, test_path: str, labels=None,
               dev_ratio=0.15, seed=42, params=None, *, purpose: str) -> dict:
        """校验输入并创建任务，交给线程池异步训练和测试。

        Args:
            model: 模型类型标识，如 rf、fasttext 或 bert。
            train_path: 训练数据集的本地文件或目录路径。
            test_path: 独立测试数据集的本地文件或目录路径。
            labels: 原始 label 到类别名称的映射字典，兼容内部调用的类别名称列表。
            dev_ratio: 从训练样本中划出验证集的比例。
            seed: 用于数据划分或训练的随机种子。
            params: 训练参数字典；缺省字段使用模型默认配置。
            purpose: 训练用途标识或已知中文用途；用于生成名称及保存用途。

        Returns:
            提交后的任务字典，包含任务 id 和当时状态。

        Raises:
            ValueError: 准备阶段的用途或路径校验未通过。
        """
        with _log_call("train.service.TrainService.submit", locals()):
            job, options = self._prepare(model, train_path, test_path, labels,
                                         dev_ratio, seed, params, purpose)
            logger.info("提交任务 job=%s model=%s name=%s", job.id, model, job.name)
            self._executor.submit(self._execute, job.id, options)
            return job.to_dict(with_log=True)

    def run_sync(self, model: str, train_path: str, test_path: str, labels=None,
                 dev_ratio=0.15, seed=42, params=None, *, purpose: str) -> dict:
        """在当前线程执行数据准备、训练、测试和模型登记。

        Args:
            model: 模型类型标识，如 rf、fasttext 或 bert。
            train_path: 训练数据集的本地文件或目录路径。
            test_path: 独立测试数据集的本地文件或目录路径。
            labels: 原始 label 到类别名称的映射字典，兼容内部调用的类别名称列表。
            dev_ratio: 从训练样本中划出验证集的比例。
            seed: 用于数据划分或训练的随机种子。
            params: 训练参数字典；缺省字段使用模型默认配置。
            purpose: 训练用途标识或已知中文用途；用于生成名称及保存用途。

        Returns:
            最终任务字典；执行失败原因记录在 error 中。

        Raises:
            ValueError: 准备阶段的用途或路径校验未通过；执行阶段错误记录在任务中。
        """
        with _log_call("train.service.TrainService.run_sync", locals()):
            job, options = self._prepare(model, train_path, test_path, labels,
                                         dev_ratio, seed, params, purpose)
            self._execute(job.id, options)
            return self.get(job.id)

    def get(self, job_id: str) -> dict:
        """在线程锁保护下读取指定任务及完整日志。

        Args:
            job_id: 训练任务的唯一 id。

        Returns:
            任务字典；不存在时抛出 JobNotFound。

        Raises:
            JobNotFound: 当前进程中不存在指定任务。
        """
        with _log_call("train.service.TrainService.get", locals()):
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    logger.warning("任务不存在 job=%s", job_id)
                    raise JobNotFound(job_id)
                return job.to_dict(with_log=True)

    def list(self) -> List[dict]:
        """在线程锁保护下列出当前进程内的任务。

        Returns:
            按创建时间倒序的任务字典列表，不含日志。
        """
        with _log_call("train.service.TrainService.list", locals()):
            with self._lock:
                jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
                return [job.to_dict() for job in jobs]

    def artifacts(self, job_id: str) -> List[dict]:
        """扫描指定任务的模型目录，列出其中的文件。

        Args:
            job_id: 训练任务的唯一 id。

        Returns:
            包含绝对路径和字节大小的字典列表；不存在的任务抛出 JobNotFound。

        Raises:
            JobNotFound: 当前进程中不存在指定任务。
        """
        with _log_call("train.service.TrainService.artifacts", locals()):
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    raise JobNotFound(job_id)
                target = output_dir_for(job.model) / job.name
            return [{"path": str(path), "size": path.stat().st_size}
                    for path in sorted(target.rglob("*")) if path.is_file()]

    # ---------- 准备阶段（校验 + 命名）----------

    def _prepare(self, model, train_path, test_path, labels, dev_ratio,
                 seed, params, purpose) -> tuple[Job, dict]:
        """校验用途和输入路径，合并训练参数，生成模型名称并建立任务。

        Args:
            model: 模型类型标识，如 rf、fasttext 或 bert。
            train_path: 训练数据集的本地文件或目录路径。
            test_path: 独立测试数据集的本地文件或目录路径。
            labels: 原始 label 到类别名称的映射字典，兼容内部调用的类别名称列表。
            dev_ratio: 从训练样本中划出验证集的比例。
            seed: 用于数据划分或训练的随机种子。
            params: 训练参数字典；缺省字段使用模型默认配置。
            purpose: 训练用途标识或已知中文用途；用于生成名称及保存用途。

        Returns:
            二元组 (Job 对象, 执行选项字典)；非法输入抛出 ValueError。

        Raises:
            ValueError: 用途无效、数据集路径或所需基座模型不存在。
        """
        with _log_call("train.service.TrainService._prepare", locals()):
            purpose = purpose_code(purpose)
            for label, path in (("训练集", train_path), ("测试集", test_path)):
                if not path or not Path(path).exists():
                    logger.warning("%s路径不存在 path=%s", label, path)
                    raise ValueError(f"{label}路径不存在: {path}")

            params = dict(params or {})
            if model == "bert":
                base = resolve_base_model(model, params.get("base_model"))
                if not Path(base).exists():
                    logger.warning("基座模型不存在 path=%s", base)
                    raise ValueError(f"基座模型不存在: {base}")
                params["base_model"] = base

            # 前端没传的 epochs / batch_size 等，用 config 里的默认值补齐
            params = merge_params(model, params)
            if model == 'rf':
                from .configs import get_config
                cfg = get_config('rf', params)
                dev_ratio, seed = cfg.train_validation_size, cfg.random_state
            logger.info("训练参数 model=%s params=%s", model, params)

            target_root = output_dir_for(model)
            name = new_model_name(model, purpose,
                                  taken=lambda candidate: (target_root / candidate).exists(),
                                  tokenizer=params.get("tokenizer", "char"),
                                  autotune=params.get("autotune", False))
            logger.info("分配模型名 name=%s type=%s purpose=%s", name.text, model, name.purpose)

            job = Job(id=uuid.uuid4().hex[:12], model=model,
                      source=str(train_path), test_source=str(test_path), name=name.text,
                      purpose=purpose)
            with self._lock:
                self._jobs[job.id] = job
            options = dict(labels=labels, dev_ratio=dev_ratio, seed=seed,
                           test_path=str(test_path), params=params)
            return job, options

    # ---------- 执行：三个阶段 ----------

    @contextmanager
    def _stage(self, name: str, job: Job):
        """为一个训练阶段记录开始日志及耗时，并将阶段异常包装为 StageError。

        Args:
            name: 当前训练阶段名称。
            job: 训练任务对象，提供名称、类型、用途、数据来源及时间等信息。

        Yields:
            上下文管理器，进入时提供追加阶段日志的 note 回调。

        Raises:
            StageError: 阶段内的执行发生异常，包装原始异常并保留阶段名。
        """
        with _log_call("train.service.TrainService._stage", locals()):
            started = time.perf_counter()
            job.trace(f"{name} 开始")
            logger.info("阶段开始 job=%s stage=%s", job.id, name)

            def note(message: str) -> None:
                """向当前任务追加一条缩进的阶段日志。

                Args:
                    message: 要记录的日志或警告文本。

                Returns:
                    None。
                """
                job.trace(f"  {message}")

            try:
                yield note
            except Exception as exc:
                job.trace(f"{name} 失败: {type(exc).__name__}: {exc}")
                raise StageError(name, exc) from exc
            finally:
                elapsed = time.perf_counter() - started
                job.stages[name] = round(elapsed, 2)
                logger.info("阶段结束 job=%s stage=%s 耗时=%.2fs", job.id, name, elapsed)

    def _execute(self, job_id: str, options: dict) -> None:
        """依次准备数据、创建训练器、训练和测试，成功后登记模型并更新任务状态。

        Args:
            job_id: 训练任务的唯一 id。
            options: 准备阶段生成的执行选项，含标签、划分比例、种子、测试路径和训练参数。

        Returns:
            None；执行异常写入任务状态和日志，最终保存完成时间。
        """
        with training_log(self._jobs[job_id].name), _log_call("train.service.TrainService._execute", locals()):
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = _now()
            self._save(job)
            work_dir = self._job_dir(job.id)
            logger.info("任务开始 job=%s model=%s name=%s work=%s",
                        job.id, job.model, job.name, work_dir)

            try:
                # 1) 处理数据集
                with self._stage(STAGE_DATA, job) as note:
                    manifest = prepare(
                        input_path=job.source, use=job.model,
                        out_dir=work_dir / "prepared",
                        labels=options["labels"], dev_ratio=options["dev_ratio"],
                        seed=options["seed"], verbose=False,
                        test_input=options["test_path"],
                        word_level=options["params"].get("tokenizer", "char") == "word",
                        tfidf_max_features=options['params'].get('tfidf_max_features'),
                    )
                    counts = "，".join(f"{ {'train': '训练集', 'dev': '验证集', 'test': '测试集'}.get(key, key)} {value} 条" for key, value in manifest['rows'].items())
                    note(f"数据已就绪：{counts}；共 {len(manifest['classes'])} 个类别")
                    data_dir = work_dir / "prepared" / job.model

                def progress(record):
                    if "iteration" in record:
                        job.progress = {"iteration": record["iteration"],
                                        "total_iterations": record["total_iterations"]}
                    elif "step" in record and "total_steps" in record:
                        job.progress = {"iteration": record["step"],
                                        "total_iterations": record["total_steps"]}
                    elif "progress_percent" in record and job.progress.get("phase") != "autotune":
                        job.progress = {"percent": record["progress_percent"]}
                    elif record.get("stage") == "autotune_started":
                        job.progress = {"phase": "autotune"}
                    if record.get("selected") or record.get("stage") in ("best_checkpoint", "completed"):
                        job.progress = {**job.progress, "finished": True}
                    job.trace(format_progress(record))
                    if any(split in record for split in ("train", "dev", "test")):
                        job.history.append(record)
                    if record.get("selected") or record.get("stage") in ("best_checkpoint", "completed"):
                        job.metrics = {key: record[key] for key in ("train", "dev", "test", "selected_epoch") if key in record}

                spec = TrainerSpec(
                    kind=job.model,
                    data_dir=data_dir,
                    model_dir=output_dir_for(job.model) / job.name,
                    base_model=resolve_base_model(job.model, options["params"].get("base_model")),
                    params=options["params"],
                    on_progress=progress,
                )

                # 2) 按模型类型路由
                with self._stage(STAGE_ROUTE, job) as note:
                    trainer = create_trainer(spec)
                    note(f"模型：{job.name}\n模型类型：{job.model}\n模型保存位置：{spec.model_dir}\n训练日志：logs/train/{job.name}/{job.name}_log.log")

                # 3) 训练
                with self._stage(STAGE_TRAIN, job) as note:
                    result = trainer.run()
                    if job.model == "fasttext":
                        options["params"].update(result.metrics.get("parameters", {}))
                    for warning in result.warnings:
                        job.trace(f"  警告: {warning}")
                    note(format_progress({"stage": "completed", **result.metrics}))

                # 4) 训练完立刻用测试集验证，前端直接拿四个指标
                with self._stage(STAGE_TEST, job) as note:
                    evaluation = evaluate(spec.model_dir, data_dir)
                    job.metrics = {
                        **evaluation["scores"],
                        "evaluated_rows": evaluation["rows"],
                        **result.metrics,
                    }
                    note(format_progress({"stage": "completed", "test": evaluation["scores"]}))

                model_dir = spec.model_dir
                evaluation["training_metrics"] = result.metrics
                if isinstance(options["labels"], dict):
                    model_dir.mkdir(parents=True, exist_ok=True)
                    (model_dir / "label_mapping.json").write_text(
                        json.dumps(options["labels"], ensure_ascii=False, indent=2), encoding="utf-8")
                job.artifacts = {str(path.relative_to(model_dir)): path.stat().st_size
                                 for path in sorted(model_dir.rglob("*")) if path.is_file()}
                job.finished_at = _now()
                if job.model == "rf":
                    shutil.copy2(data_dir / "tfidf_vectorizer.joblib", model_dir / "tfidf_vectorizer.joblib")
                self.model_store.register(job=job, model_path=model_dir,
                                          params=options["params"], evaluation=evaluation,
                                          label_mapping=manifest["classes"])
                job.trace("模型和标签映射已保存并登记到模型管理表")
                job.status = "succeeded"
                logger.info("任务成功 job=%s metrics=%s", job.id, job.metrics)
            except StageError as error:
                job.status = "failed"
                job.error = str(error)
                logger.exception("任务失败 job=%s stage=%s model=%s",
                                 job.id, error.stage, job.model)
            except Exception as exc:  # noqa: BLE001 - 兜底，任何异常都要落回任务状态
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                logger.exception("任务异常 job=%s model=%s", job.id, job.model)
            finally:
                job.finished_at = _now()
                self._save(job)
                logger.info("任务结束 job=%s status=%s 各阶段=%s",
                            job.id, job.status, job.stages)

    # ---------- 内部工具 ----------

    def _save(self, job: Job) -> None:
        """在线程锁保护下将任务对象写入内存任务表。

        Args:
            job: 训练任务对象，提供名称、类型、用途、数据来源及时间等信息。

        Returns:
            None。
        """
        with _log_call("train.service.TrainService._save", locals()):
            with self._lock:
                self._jobs[job.id] = job

    def _job_dir(self, job_id: str) -> Path:
        """获取任务工作目录，并创建缺失的目录。

        Args:
            job_id: 训练任务的唯一 id。

        Returns:
            任务工作目录 Path。
        """
        directory = self.work_root / job_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory
