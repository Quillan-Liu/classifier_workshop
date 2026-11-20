"""生成 API 的 OpenAPI JSON，并启动训练服务。

    python start.py                # 先导出 openapi.json，再启动服务
    python start.py --no-serve     # 只导出 JSON
    python start.py --no-export    # 只启动服务
    python start.py --port 8007    # 临时换端口

服务起来后：
    http://127.0.0.1:8008/docs     Swagger 调试页
    http://127.0.0.1:8008/redoc    Redoc 文档
"""

import argparse
import json
from pathlib import Path

import uvicorn

from train.api import app
from train.config import OPENAPI_PATH, SERVER_HOST, SERVER_PORT
from train.logging import get_logger

logger = get_logger(__name__)


def _downgrade_nullable(node) -> None:
    """把 FastAPI 输出的 3.1 写法降级成 OpenAPI 3.0.2，Apifox 才解析得完整。

    - anyOf [X, null] -> X + nullable: true（3.0 没有 null 类型）
    - contentMediaType -> format: binary（3.0 没有该关键字，文件字段会退化成文本框）
    """
    if isinstance(node, dict):
        if "contentMediaType" in node:
            node.pop("contentMediaType")
            node["format"] = "binary"
        variants = node.get("anyOf")
        if isinstance(variants, list):
            rest = [item for item in variants if item != {"type": "null"}]
            if len(rest) == 1 and len(rest) != len(variants) and isinstance(rest[0], dict):
                node.pop("anyOf")
                node.update(rest[0])
                node["nullable"] = True
        for value in node.values():
            _downgrade_nullable(value)
    elif isinstance(node, list):
        for value in node:
            _downgrade_nullable(value)


def export_openapi(path: Path = OPENAPI_PATH, host: str = SERVER_HOST,
                   port: int = SERVER_PORT) -> Path:
    """把 FastAPI 的 OpenAPI 描述写成 JSON 文件（给 Apifox/前端生成客户端用）。

    固定导出 OpenAPI 3.0.2：Apifox 对 3.0 的解析最完整，3.1 会丢枚举/可空字段。
    """
    app.openapi_version = "3.0.2"
    schema = app.openapi()
    _downgrade_nullable(schema)
    # 跟随访问地址，局域网客户端调试时不会请求自己的 localhost。
    schema["servers"] = [{"url": "/", "description": "当前服务"}]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    routes = [f"{method.upper():6} {route}"
              for route, operations in schema["paths"].items() for method in operations]
    logger.info("OpenAPI 已导出: %s", path)
    logger.info("共 %s 个接口:", len(routes))
    for item in routes:
        logger.info("  %s", item)
    return path


def serve(host: str = SERVER_HOST, port: int = SERVER_PORT) -> None:
    logger.info("启动训练服务: http://%s:%s", host, port)
    logger.info("接口文档: http://%s:%s/docs", host, port)
    uvicorn.run(app, host=host, port=port, log_config=None)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="生成 API JSON 并启动训练服务")
    parser.add_argument("--host", default=SERVER_HOST, help="监听地址")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="监听端口")
    parser.add_argument("--no-serve", action="store_true", help="只导出 OpenAPI，不启动服务")
    parser.add_argument("--no-export", action="store_true", help="不导出 OpenAPI，只启动服务")
    parser.add_argument("--out", default=str(OPENAPI_PATH), help="OpenAPI 输出路径")
    args = parser.parse_args(argv)

    if not args.no_export:
        export_openapi(Path(args.out))
    if not args.no_serve:
        serve(args.host, args.port)


if __name__ == "__main__":
    main()
