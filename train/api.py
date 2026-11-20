"""接口层：提交训练集、测试集、label 映射文件及模型类型，返回测试结果。

日志是全局的，整个进程只在这里 setup 一次，其它模块只 get_logger。
"""

from .logging import log_call as _log_call

from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool
from inference.api import router as inference_router
from inference.cache import catalog

import json
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .configs import MODEL_OPTIONS, ModelKind
from .config import PROJECT_DIR, defaults_for
from .logging import get_logger, setup_logging

# 用 Literal 而不是 Enum：OpenAPI 里会内联成 enum 数组，Apifox 才能渲染成下拉选项表
ModelKindLiteral = Literal["rf", "fasttext", "bert"]
from .service import JobNotFound, TrainService
from .dataset.label_mapping import load_label_mapping
from model_management.api import router as model_router, store as model_store
from model_management.purposes import Purpose

setup_logging()
logger = get_logger(__name__)

service = TrainService(model_store=model_store)
@asynccontextmanager
async def lifespan(app):
    await run_in_threadpool(catalog.start)
    try:
        yield
    finally:
        await run_in_threadpool(catalog.stop)


app = FastAPI(title="TMF 模型工作台", version="2.0.0", lifespan=lifespan)
app.include_router(inference_router)
app.include_router(model_router)
app.mount("/ui", StaticFiles(directory=PROJECT_DIR / "frontend"), name="frontend")


@app.get("/training", include_in_schema=False)
def training_page():
    """返回训练页面的 HTML 文件响应。

    Returns:
        训练页面的 FileResponse。
    """
    with _log_call("train.api.training_page", locals()):
        return FileResponse(PROJECT_DIR / "frontend" / "training" / "index.html")


@app.get("/models/{model}/defaults")
def training_defaults(model: ModelKind):
    """查询指定模型类型的默认训练参数。

    Args:
        model: 模型类型标识，如 rf、fasttext 或 bert。

    Returns:
        包含 model 和 params 的字典。
    """
    with _log_call("train.api.training_defaults", locals()):
        return {"model": model.value, "params": defaults_for(model.value)}


def _save_upload(upload: UploadFile, target_dir: Path, tag: str) -> Path:
    """保存上传文件；ZIP 文件会解压，返回数据集输入路径。

    Args:
        upload: 上传文件对象，包含文件名及可读文件流。
        target_dir: 上传文件和解压文件的保存目录。
        tag: 保存文件的前缀，用于区分训练集和测试集。

    Returns:
        保存的文件路径，或 ZIP 解压目录。
    """
    with _log_call("train.api._save_upload", locals()):
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / f"{tag}_{Path(upload.filename or 'dataset.bin').name}"
        with dest.open("wb") as file:
            shutil.copyfileobj(upload.file, file)
        logger.info("上传文件已保存 tag=%s path=%s size=%s", tag, dest, dest.stat().st_size)

        if dest.suffix.lower() == ".zip":
            extract_dir = target_dir / f"{tag}_unzipped"
            with zipfile.ZipFile(dest) as archive:
                archive.extractall(extract_dir)
            logger.info("压缩包已解包 tag=%s dir=%s", tag, extract_dir)
            return extract_dir
        return dest


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """执行下游请求处理，记录请求方法、路径、状态码和耗时。

    Args:
        request: 当前 HTTP 请求对象。
        call_next: 执行后续中间件及路由的异步回调。

    Returns:
        下游处理器生成的 HTTP 响应；异常记录后继续抛出。
    """
    with _log_call("train.api.log_requests", locals()):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("请求处理异常 method=%s path=%s", request.method, request.url.path)
            raise
        logger.info("%s %s -> %s (%.0f ms)", request.method, request.url.path,
                    response.status_code, (time.perf_counter() - started) * 1000)
        return response


@app.exception_handler(JobNotFound)
async def handle_not_found(request: Request, exc: JobNotFound):
    """将任务不存在异常转换为 HTTP 404，并记录警告日志。

    Args:
        request: 当前 HTTP 请求对象。
        exc: 需要转换为 HTTP 响应的异常对象。

    Returns:
        包含 detail 的 JSONResponse。
    """
    with _log_call("train.api.handle_not_found", locals()):
        logger.warning("任务不存在 %s %s", request.method, request.url.path)
        return JSONResponse(status_code=404, content={"detail": f"任务不存在: {exc}"})


@app.exception_handler(ValueError)
async def handle_value_error(request: Request, exc: ValueError):
    """将业务参数异常转换为 HTTP 400，并记录警告日志。

    Args:
        request: 当前 HTTP 请求对象。
        exc: 需要转换为 HTTP 响应的异常对象。

    Returns:
        包含 detail 的 JSONResponse。
    """
    with _log_call("train.api.handle_value_error", locals()):
        logger.warning("参数错误 %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception):
    """记录未处理异常的堆栈并返回 HTTP 500。

    Args:
        request: 当前 HTTP 请求对象。
        exc: 需要转换为 HTTP 响应的异常对象。

    Returns:
        包含异常类型和原因的 JSONResponse。
    """
    with _log_call("train.api.handle_unexpected", locals()):
        logger.exception("未处理异常 %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


@app.get("/health")
def health():
    """提供服务健康检查。

    Returns:
        固定字典 {"status": "ok"}。
    """
    with _log_call("train.api.health", locals()):
        return {"status": "ok"}


@app.get("/models")
def models():
    """提供可训练模型类型及其前端展示选项。

    Returns:
        包含 models 标识列表和 options 选项列表的字典。
    """
    with _log_call("train.api.models", locals()):
        return {
            "models": [option["value"] for option in MODEL_OPTIONS],
            "options": MODEL_OPTIONS,
        }


@app.post("/train")
def train(
    train_data: UploadFile = File(..., description="训练集文件（.arrow/.jsonl/txt/.zip）"),
    test_data: UploadFile = File(..., description="测试集文件（同上）"),
    label_mapping: UploadFile = File(..., description='必填：label 到类别名的 JSON 对象，或每行 label<TAB>类别名的 TXT/TSV'),
    model: ModelKindLiteral = Form(..., description="训练模型类型，只能是 rf / fasttext / bert"),
    epochs: int | None = Form(None, description="选填，不传用 config 默认"),
    batch_size: int | None = Form(None, description="选填，不传用 config 默认"),
    purpose: Purpose = Form(..., description="必填：训练用途英文标识，GET /model-registry-options 获取中英文选项；用于模型命名"),
    params: str | None = Form(None, description="额外超参 JSON"),
    dev_ratio: float = Form(0.15, description="从训练集切验证集的比例"),
    sync: bool = Form(False, description="true=同步等结果，false=返回任务 id"),
):
    """保存训练集和测试集，提交训练任务；sync=true 时等待训练和测试完成。

    Args:
        train_data: 上传的训练数据文件。
        test_data: 上传的独立测试数据文件。
        label_mapping: 必填的原始 label 到类别名称映射文件。
        model: 模型类型标识，如 rf、fasttext 或 bert。
        epochs: 可选训练轮数，覆盖默认配置。
        batch_size: 可选训练批大小，覆盖默认配置。
        purpose: 训练用途标识或已知中文用途；用于生成名称及保存用途。
        params: 可选额外超参数 JSON 字符串。
        dev_ratio: 从训练样本中划出验证集的比例。
        sync: 是否同步等待训练和测试完成。

    Returns:
        异步返回任务信息；同步返回包含状态、测试指标及错误信息的最终任务字典。

    Raises:
        ValueError: 用途、输入路径或训练参数非法；由接口异常处理器转换为 400。
    """
    with _log_call("train.api.train", locals()):
        logger.info("收到训练请求 model=%s train=%s test=%s sync=%s",
                    model, train_data.filename, test_data.filename, sync)

        work_dir = service.work_root / "uploads" / uuid.uuid4().hex[:8]
        train_path = _save_upload(train_data, work_dir, "train")
        test_path = _save_upload(test_data, work_dir, "test")
        if Path(label_mapping.filename or '').suffix.lower() not in ('.json', '.txt', '.tsv'):
            raise ValueError("映射文件仅支持 JSON、TXT 或 TSV")
        mapping_path = _save_upload(label_mapping, work_dir, "label_mapping")
        label_list = load_label_mapping(mapping_path)

        try:
            extra = json.loads(params) if params else {}
        except json.JSONDecodeError as exc:
            logger.warning("params 不是合法 JSON: %s", exc)
            raise ValueError(f"params 不是合法 JSON: {exc}") from exc
        if epochs is not None:
            extra["epochs"] = epochs
        if batch_size is not None:
            extra["batch_size"] = batch_size

        logger.info("已读取 label 映射，类别数量=%s", len(label_list))

        if sync:
            return service.run_sync(model=model, train_path=str(train_path),
                                    test_path=str(test_path), labels=label_list,
                                    dev_ratio=dev_ratio, params=extra, purpose=purpose)
        return service.submit(model=model, train_path=str(train_path),
                              test_path=str(test_path), labels=label_list,
                              dev_ratio=dev_ratio, params=extra, purpose=purpose)


@app.get("/jobs")
def list_jobs():
    """查询当前进程内的训练任务，按创建时间倒序返回。

    Returns:
        包含 jobs 列表的字典；服务重启后内存任务不保留。
    """
    with _log_call("train.api.list_jobs", locals()):
        return {"jobs": service.list()}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """查询指定训练任务的状态、阶段耗时、指标及日志。

    Args:
        job_id: 训练任务的唯一 id。

    Returns:
        完整任务字典；任务不存在时由异常处理器返回 404。
    """
    with _log_call("train.api.get_job", locals()):
        return service.get(job_id)


@app.get("/jobs/{job_id}/artifacts")
def get_artifacts(job_id: str):
    """查询训练任务的模型文件路径及文件大小。

    Args:
        job_id: 训练任务的唯一 id。

    Returns:
        包含 job_id、status 和 files 的字典。
    """
    with _log_call("train.api.get_artifacts", locals()):
        job = service.get(job_id)
        return {"job_id": job["id"], "status": job["status"],
                "files": service.artifacts(job_id)}
