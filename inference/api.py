"""独立推理路由。"""
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from model_management.purposes import MODEL_TYPE_LABELS, PURPOSE_EN_TO_ZH
from .cache import catalog
from .service import predict
from .chart import probability_chart

router = APIRouter(tags=["推理"])


class ChartItem(BaseModel):
    """图表单个标签及其模型输出概率。"""
    class_name: str = Field(min_length=1, max_length=200)
    probability: float = Field(ge=0, le=1, allow_inf_nan=False)


class ChartRequest(BaseModel):
    """需要绘制的全部标签。"""
    items: list[ChartItem] = Field(min_length=1, max_length=500)


@router.post("/inference/chart", response_class=Response)
def chart(request: ChartRequest):
    """根据标签概率生成 SVG；所有标签均参与绘制，名称按 XML 转义。"""
    return Response(probability_chart([item.model_dump() for item in request.items]),
                    media_type="image/svg+xml", headers={"Cache-Control": "no-store"})

@router.get("/inference", include_in_schema=False)
def page():
    """返回推理页面。

    Returns:
        FileResponse: 项目内的推理前端 HTML 文件。
    """
    return FileResponse(Path(__file__).resolve().parents[1] / "frontend/inference/index.html")

@router.get("/inference/models")
def models():
    """查询可推理模型及筛选选项。

    Returns:
        dict: 包含模型列表 ``items``、用途 ``purposes`` 和模型类型
        ``model_types``；模型列表隐藏内部路径及训练参数。
    """
    items = catalog.get()
    return {"items": [{key: value for key, value in item.items() if key not in {"path", "params"}} for item in items],
            "purposes": [{"value": code, "label": PURPOSE_EN_TO_ZH.get(code, code)} for code in dict.fromkeys(item["purpose"] for item in items)],
            "model_types": MODEL_TYPE_LABELS}

class PredictRequest(BaseModel):
    """单条文本分类请求，拒绝未声明的字段。

    Attributes:
        model_id (str): 登记模型 ID，长度为 1 到 200。
        purpose (str): 用途标识，须与模型登记用途一致，长度为 1 到 200。
        text (str): 待分类文本，长度为 1 到 20000，不允许仅包含空白。
    """
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=200)
    purpose: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=20000)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value):
        """校验文本包含非空白字符，保留原始文本。

        Args:
            value (str): 已通过字段类型和长度检查的文本。

        Returns:
            str: 未修改的输入文本。

        Raises:
            ValueError: 输入仅包含空白字符，由请求校验转换为 HTTP 422。
        """
        if not value.strip():
            raise ValueError("请输入非空字符串")
        return value

@router.post("/inference/predict")
def infer(request: PredictRequest):
    """查询模型目录，校验用途后执行分类。

    Args:
        request (PredictRequest): 已校验的模型 ID、用途和文本。

    Returns:
        dict: 模型标识、名称、预测标签及类别名称。

    Raises:
        HTTPException: 模型不存在时返回 404，用途不匹配时返回 400，
            模型文件缺失或不可读取时返回 409。
    """
    record = next((item for item in catalog.get() if item["id"] == request.model_id), None)
    if record is None:
        raise HTTPException(404, "模型不存在或已删除")
    if record["purpose"] != request.purpose:
        raise HTTPException(400, "模型与所选用途不匹配")
    try:
        return predict(record, request.text)
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(409, "模型产物不完整或不可读取") from exc
