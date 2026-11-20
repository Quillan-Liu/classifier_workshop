"""训练服务入口。

启动：
    uvicorn service.main:app --host 127.0.0.1 --port 8006

流程：提交任务 -> 202 + job_id -> 轮询 /jobs/{job_id} 拿状态和指标。
"""

import json
import shutil
import zipfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile

from .jobs import JOB_STORE
from .schemas import ModelType, TrainRequest
from train.logging import setup_logging

setup_logging()

app = FastAPI(title="TMF 训练服务", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return {"models": ["rf", "fasttext", "bert"]}


@app.post("/train", status_code=202)
def train(request: TrainRequest, background: BackgroundTasks):
    """用服务器本地已有的数据集路径提交训练。"""
    source = Path(request.dataset_path)
    if not source.exists():
        raise HTTPException(status_code=400, detail=f"数据集路径不存在: {source}")
    job = JOB_STORE.create(model=request.model.value, source=str(source))
    background.add_task(JOB_STORE.run, job.id, request)
    return job.to_dict(with_log=True)


@app.post("/train/upload", status_code=202)
def train_upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    model: ModelType = Form(...),
    labels: str | None = Form(None),
    params: str | None = Form(None),
    dev_ratio: float = Form(0.15),
    test_ratio: float = Form(0.15),
    seed: int = Form(42),
):
    """上传数据集文件（.arrow/.jsonl/.txt/.csv 或它们的 .zip）后提交训练。"""
    job = JOB_STORE.create(model=model.value, source=file.filename or "upload")
    raw_dir = JOB_STORE.job_dir(job.id) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / Path(file.filename or "dataset.bin").name

    with dest.open("wb") as target:
        shutil.copyfileobj(file.file, target)

    data_path = dest
    if dest.suffix.lower() == ".zip":
        extract_dir = raw_dir / "unzipped"
        with zipfile.ZipFile(dest) as archive:
            archive.extractall(extract_dir)
        data_path = extract_dir

    request = TrainRequest(
        model=model,
        dataset_path=str(data_path),
        labels=[item.strip() for item in labels.split(",")] if labels else None,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
        params=json.loads(params) if params else {},
    )
    background.add_task(JOB_STORE.run, job.id, request)
    return job.to_dict(with_log=True)


@app.get("/jobs")
def list_jobs():
    return {"jobs": JOB_STORE.list()}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return JOB_STORE.get(job_id).to_dict(with_log=True)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")


@app.get("/jobs/{job_id}/artifacts")
def get_artifacts(job_id: str):
    try:
        job = JOB_STORE.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
    model_dir = JOB_STORE.job_dir(job_id) / "model"
    files = [{"path": str(path), "size": path.stat().st_size}
             for path in sorted(model_dir.rglob("*")) if path.is_file()]
    return {"job_id": job.id, "status": job.status, "files": files}
