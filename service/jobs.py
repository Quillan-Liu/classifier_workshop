"""训练任务存储与执行。第一步先用进程内字典，重启即丢。"""

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from train.dataset import prepare

from .trainers import TRAINERS

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    model: str
    source: str
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    metrics: dict | None = None
    artifacts: dict | None = None
    error: str | None = None
    log: list = field(default_factory=list)

    def add(self, message):
        self.log.append(f"[{_now()}] {message}")

    def to_dict(self, with_log=False):
        data = {
            "id": self.id,
            "model": self.model,
            "source": self.source,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "error": self.error,
        }
        if with_log:
            data["log"] = self.log
        return data


class JobStore:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def create(self, model, source):
        job = Job(id=uuid.uuid4().hex[:12], model=model, source=source)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def list(self):
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)
        return [job.to_dict() for job in jobs]

    def job_dir(self, job_id):
        path = RUNS_DIR / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run(self, job_id, request):
        """在 FastAPI 的后台任务里同步执行：准备数据 -> 训练 -> 记录结果。"""
        job = self.get(job_id)
        job.status = "running"
        job.started_at = _now()
        root = self.job_dir(job_id)
        try:
            job.add(f"准备数据: {request.dataset_path}")
            manifest = prepare(
                input_path=request.dataset_path,
                use=request.model,
                out_dir=root / "prepared",
                labels=request.labels,
                dev_ratio=request.dev_ratio,
                test_ratio=request.test_ratio,
                seed=request.seed,
                verbose=False,
            )
            job.add(f"数据就绪 rows={manifest['rows']} classes={len(manifest['classes'])}")

            trainer = TRAINERS[request.model]
            job.metrics = trainer(root / "prepared" / request.model, root / "model",
                                  request.params, job.add)
            job.add("训练完成")

            model_dir = root / "model"
            job.artifacts = {
                str(path.relative_to(model_dir)): path.stat().st_size
                for path in sorted(model_dir.rglob("*")) if path.is_file()
            }
            job.status = "succeeded"
        except Exception as exc:  # noqa: BLE001 - 任何失败都要落回任务状态
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.add("失败: " + traceback.format_exc(limit=6))
        finally:
            job.finished_at = _now()


JOB_STORE = JobStore()
