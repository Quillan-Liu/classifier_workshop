"""训练服务的请求/响应模型。"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    rf = "rf"
    fasttext = "fasttext"
    bert = "bert"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class TrainRequest(BaseModel):
    """训练请求。dataset_path 是服务器本地可访问的路径。"""

    model: ModelType
    dataset_path: str
    labels: Optional[List[str]] = None
    dev_ratio: float = Field(0.15, gt=0, lt=1)
    test_ratio: float = Field(0.15, gt=0, lt=1)
    seed: int = 42
    params: Dict[str, Any] = Field(default_factory=dict)
