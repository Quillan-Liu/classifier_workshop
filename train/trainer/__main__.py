"""命令行入口：

    python -m train.trainer --kind rf --data-dir data/intent/rf --model-dir runs/demo/model
    python -m train.trainer --kind bert --data-dir data/intent/bert --model-dir runs/demo/model \
        --base-model D:/models/bert-base-chinese --params '{"epochs":3}'
"""

from ..logging import log_call as _log_call

import argparse
import json
import sys
from pathlib import Path

from ..logging import get_logger, setup_logging

from .registry import supported, train
from .spec import TrainerError, TrainerSpec

logger = get_logger(__name__)


def build_parser():
    """构建训练命令行参数，限制类型为已注册训练器。

    Returns:
        argparse.ArgumentParser 实例。
    """
    with _log_call("train.trainer.__main__.build_parser", locals()):
        parser = argparse.ArgumentParser(prog="trainer", description="按类型训练模型")
        parser.add_argument("--kind", required=True, choices=supported(), help="训练类型")
        parser.add_argument("--data-dir", required=True, help="dataset_tool 产出的目录")
        parser.add_argument("--model-dir", required=True, help="模型输出目录")
        parser.add_argument("--base-model", default=None, help="基座模型路径或名称（bert 必填）")
        parser.add_argument("--params", default="{}", help="超参 JSON")
        return parser


def main(argv=None):
    """解析参数、建立 TrainerSpec、执行训练并打印 JSON 结果。

    Args:
        argv: 命令行参数列表；None 表示读取当前进程参数。

    Returns:
        训练完成后的 TrainResult。
    """
    with _log_call("train.trainer.__main__.main", locals()):
        args = build_parser().parse_args(argv)
        setup_logging()
        spec = TrainerSpec(
            kind=args.kind,
            data_dir=Path(args.data_dir),
            model_dir=Path(args.model_dir),
            base_model=args.base_model,
            params=json.loads(args.params),
        )
        result = train(spec)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return result


if __name__ == "__main__":
    try:
        main()
    except TrainerError as error:
        logger.error("训练失败: %s", error)
        sys.exit(2)
