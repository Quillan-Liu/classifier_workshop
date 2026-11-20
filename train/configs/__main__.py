"""python -m train.configs —— 打印三类模型的默认训练参数。"""

from ..logging import log_call as _log_call

import json

from . import CONFIGS, defaults


def main():
    """依次将已注册模型类型的默认训练配置打印到终端。

    Returns:
        None。
    """
    with _log_call("train.configs.__main__.main", locals()):
        for model in CONFIGS:
            print(f"{model:9}", json.dumps(defaults(model), ensure_ascii=False))


if __name__ == "__main__":
    main()
