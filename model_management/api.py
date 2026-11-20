"""模型管理路由，可挂载到 FastAPI 应用。"""

from train.logging import log_call as _log_call
from pathlib import Path
import sqlite3

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .store import ModelStore
from .purposes import PURPOSE_EN_TO_ZH, PURPOSE_ZH_TO_EN, PURPOSE_OPTIONS

store = ModelStore()
router = APIRouter()


@router.get("/model-registry-options", tags=["模型管理"])
def model_options():
    """提供用途下拉选项及中英文双向映射。

    Returns:
        包含 purposes、en_to_zh、zh_to_en 的字典。
    """
    with _log_call("model_management.api.model_options", locals()):
        return {"purposes": PURPOSE_OPTIONS, "en_to_zh": PURPOSE_EN_TO_ZH,
                "zh_to_en": PURPOSE_ZH_TO_EN}


@router.get("/", include_in_schema=False)
def management_page():
    """返回模型管理页面的 HTML 文件响应。

    Returns:
        管理页面的 FileResponse。
    """
    with _log_call("model_management.api.management_page", locals()):
        return FileResponse(Path(__file__).resolve().parents[1] / "frontend" / "model_management" / "index.html")


class ModelUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(None, min_length=1, max_length=200)
    purpose: str | None = Field(None, min_length=1, max_length=200)

    @field_validator("name", "purpose", mode="before")
    @classmethod
    def validate_text(cls, value):
        """校验名称或用途必须为非空字符串，并去除首尾空白。

        Args:
            value: 待校验或转换的输入值。

        Returns:
            清理后的字符串；非法输入抛出 ValueError，由框架转换为 422。

        Raises:
            ValueError: 值不是非空字符串。
        """
        with _log_call("model_management.api.ModelUpdate.validate_text", locals()):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("必须为非空字符串")
            return value.strip()

    @model_validator(mode="after")
    def require_changes(self):
        """校验 PATCH 请求至少显式提供 name 或 purpose。

        Returns:
            已校验的 ModelUpdate；空修改抛出 ValueError。

        Raises:
            ValueError: 请求未提供任何可修改字段。
        """
        with _log_call("model_management.api.ModelUpdate.require_changes", locals()):
            if not self.model_fields_set:
                raise ValueError("至少提供 name 或 purpose")
            return self


@router.get("/model-registry", tags=["模型管理"])
def list_registered_models(page: int = Query(1, ge=1),
                           page_size: int = Query(10, ge=1, le=100)):
    """分页查询未删除的模型，列表仅携带 test 指标概要。

    Args:
        page: 从 1 开始的页码。
        page_size: 每页记录数，允许 1 至 100，默认 10。

    Returns:
        包含 items、page、page_size、total、has_more 的字典。
    """
    with _log_call("model_management.api.list_registered_models", locals()):
        return store.list(page, page_size)


class BatchDeleteRequest(BaseModel):
    """一次批量删除至多 100 个登记模型。"""
    ids: list[str] = Field(min_length=1, max_length=100)

    @field_validator("ids")
    @classmethod
    def validate_ids(cls, values):
        if any(not value.strip() or len(value) > 200 for value in values):
            raise ValueError("模型 ID 必须为 1 到 200 个字符的非空字符串")
        return list(dict.fromkeys(values))


@router.post("/model-registry/batch-delete", tags=["模型管理"])
def batch_delete_registered_models(request: BatchDeleteRequest):
    """逐项复用软删除逻辑，返回成功与失败项，部分失败不阻止后续删除。

    模型文件不会被永久删除，目录重命名和缓存刷新与单项删除一致。
    """
    return _delete_registered_models(request.ids)


@router.post("/model-registry/delete-all", tags=["模型管理"])
def delete_all_registered_models():
    """软删除请求开始时的全部可用模型，包含所有分页，不永久移除文件。"""
    return _delete_registered_models(store.list_ids())


def _delete_registered_models(model_ids):
    """共用逐项软删除流程，失败项与成功项分别返回。"""
    deleted, failed = [], []
    for model_id in model_ids:
        try:
            deleted.append(store.delete(model_id))
        except KeyError:
            failed.append({"id": model_id, "error": "模型记录不存在"})
        except (OSError, ValueError, sqlite3.Error) as exc:
            failed.append({"id": model_id, "error": str(exc)})
    return {"deleted": deleted, "failed": failed}


@router.get("/model-registry/{model_id}", tags=["模型管理"])
def get_registered_model(model_id: str):
    """按 id 查询未删除模型及完整 test 结果。

    Args:
        model_id: 模型登记记录的唯一 id。

    Returns:
        模型详情字典；模型不存在或已删除时抛出 HTTP 404。

    Raises:
        HTTPException: 模型不存在或已删除，状态码 404。
    """
    with _log_call("model_management.api.get_registered_model", locals()):
        try:
            return store.get(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="模型记录不存在")


@router.patch("/model-registry/{model_id}", tags=["模型管理"])
def update_registered_model(model_id: str, changes: ModelUpdate):
    """修改模型名称和用途，名称变更时同步重命名模型目录及更新路径。

    Args:
        model_id: 模型登记记录的唯一 id。
        changes: 待修改的 name 和/或 purpose 字段。

    Returns:
        修改后的完整模型记录；模型不可用时抛出 HTTP 404。

    Raises:
        HTTPException: 模型不存在或已删除，状态码 404。
    """
    with _log_call("model_management.api.update_registered_model", locals()):
        try:
            return store.update(model_id, changes.model_dump(exclude_unset=True))
        except KeyError:
            raise HTTPException(status_code=404, detail="模型记录不存在")


@router.delete("/model-registry/{model_id}", tags=["模型管理"])
def delete_registered_model(model_id: str):
    """软删除模型，为数据库名称及模型目录同步添加 _del_ 前缀。

    Args:
        model_id: 模型登记记录的唯一 id。

    Returns:
        包含 id 和 is_deleted=true 的字典；重复删除成功，不删除模型文件。

    Raises:
        HTTPException: 模型不存在，状态码 404。
    """
    with _log_call("model_management.api.delete_registered_model", locals()):
        try:
            return store.delete(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="模型记录不存在")


