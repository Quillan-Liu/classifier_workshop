"""彩色日志：[info] 绿 / [warning] 黄 / [err] 红。

终端里带颜色，写文件时自动去色。Windows 下先开启 ANSI 支持。
"""

import logging
import os
import sys
import time
import threading
from contextlib import contextmanager
from pathlib import Path

TAGS = {
    "DEBUG": ("[debug]", "\033[90m"),
    "INFO": ("[info]", "\033[32m"),
    "WARNING": ("[warning]", "\033[33m"),
    "ERROR": ("[err]", "\033[31m"),
    "CRITICAL": ("[err]", "\033[31m"),
}
RESET = "\033[0m"
PLAIN_FORMAT = "%(asctime)s %(tag)s [%(threadName)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
DATE_FORMAT = "%H:%M:%S"
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
_training_logs = threading.local()


def training_log_dir(model_name):
    directory = LOG_DIR / "train" / Path(model_name).name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@contextmanager
def training_log(model_name):
    """按任务线程保存完整训练日志；嵌套调用不重复添加 handler。"""
    if getattr(_training_logs, "active", False):
        yield
        return
    name = Path(model_name).name
    directory = training_log_dir(name)
    handler = logging.FileHandler(directory / f"{name}_log.log", encoding="utf-8")
    thread_id = threading.get_ident()
    handler.addFilter(lambda record: record.thread == thread_id and record.name != "train.calls")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    root.addHandler(handler)
    _training_logs.active = True
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
        _training_logs.active = False


def format_number(value, *, loss=False):
    """日志数字最多四位小数；loss 使用科学计数法，极小非零数保留数量级。"""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    if not loss and isinstance(value, int):
        return str(value)
    scientific = loss or (value != 0 and abs(value) < 0.0001)
    if scientific:
        mantissa, separator, exponent = f"{value:.4e}".partition("e")
        return mantissa.rstrip("0").rstrip(".") + separator + exponent
    return f"{value:.4f}".rstrip("0").rstrip(".")


def format_progress(record):
    """将结构化训练记录显示为可读的轮次和分组指标。"""
    stages = {"autotune_started": "开始自动调参", "native_training_started": "开始 fastText 原生训练", "native_training": "fastText 训练中", "parameters_selected": "实际训练参数", "completed": "训练完成 · 最终评估", "best_checkpoint": "验证集最佳模型评估"}
    label = f"第 {record.get('epoch', record.get('round'))} 轮" if 'epoch' in record or 'round' in record else stages.get(record.get('stage'), '训练进度')
    if record.get('stage') == 'epoch_evaluation':
        label += " · 评估结果"
    if record.get('selected'):
        label = "验证集最佳轮次模型"
    lines = [label]
    if "selected_epoch" in record:
        lines.append(f"  最终产物：第 {record['selected_epoch']} 轮训练的模型")
    for key, title in (("tree", "累计树数"), ("batch", "批次"), ("step", "步数"), ("batch_loss", "批次 Loss"), ("average_batch_loss", "平均 Loss"), ("learning_rate", "学习率"), ("native_loss", "训练 Loss"), ("progress_percent", "进度(%)")):
        if key in record:
            lines[0] += f" · {title}={format_number(record[key], loss='loss' in key)}"
    if 'duration_seconds' in record:
        lines.append(f"  搜索时长：{format_number(record['duration_seconds'])} 秒")
    if 'parameters' in record:
        titles = {"epochs": "训练轮数", "lr": "学习率", "dim": "向量维度", "word_ngrams": "词 N-gram", "bucket": "哈希桶数量", "minn": "最小字符 N-gram", "maxn": "最大字符 N-gram"}
        lines.extend(f"  {titles.get(key, key)}：{format_number(value, loss='loss' in key)}" for key, value in record['parameters'].items())
    for split, title in (("train", "训练集"), ("dev", "验证集"), ("test", "测试集")):
        if split in record:
            values = record[split]
            metrics = [f"{name}={format_number(values[key], loss=key == 'loss')}" for key, name in (("accuracy", "准确率"), ("precision", "精确率"), ("recall", "召回率"), ("f1", "F1"), ("loss", "Loss")) if key in values]
            lines.append(f"  {title}：" + "  ".join(metrics))
    return "\n".join(lines)


def _argument_summary(values):
    """保留小参数；数据集、模型等大对象仅记录类型和大小。"""
    result = {}
    for key, value in values.items():
        if key in {"self", "cls", "text", "texts", "samples", "groups"}:
            result[key] = type(value).__name__
        elif value is None or isinstance(value, (bool, int, float, str, Path)):
            result[key] = str(value)[:500] if isinstance(value, (str, Path)) else value
        elif key in {"labels", "explicit_names", "params", "options"}:
            result[key] = repr(value)[:1500]
        else:
            result[key] = f"{type(value).__name__}(size={len(value)})" if isinstance(value, (list, tuple, dict, set)) else type(value).__name__
    return result


@contextmanager
def log_call(name, arguments):
    """记录函数入口、出口及逃逸异常，不改变返回值和异常传播。"""
    logger = logging.getLogger("train.calls")
    started = time.perf_counter()
    logger.info("函数进入 %s 参数=%s", name, _argument_summary(arguments))
    try:
        yield
    except Exception:
        logger.exception("函数异常 %s 参数=%s", name, _argument_summary(arguments))
        raise
    finally:
        logger.info("函数退出 %s 耗时=%.4fs", name, time.perf_counter() - started)


def _enable_windows_ansi() -> bool:
    """尝试启用 Windows 控制台的 ANSI 颜色支持；失败时安全退回无支持状态。

    Returns:
        是否启用成功；非 Windows 平台直接返回 True。
    """
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:  # noqa: BLE001 - 开不了就不上色，不能因为日志挂掉服务
        return False


class ColorFormatter(logging.Formatter):
    def __init__(self, use_color: bool):
        """初始化日志格式器并保存是否输出 ANSI 颜色的设置。

        Args:
            use_color: 是否为格式化日志添加 ANSI 颜色。

        Returns:
            None。
        """
        super().__init__(PLAIN_FORMAT, DATE_FORMAT)
        self.use_color = use_color

    def format(self, record):
        """给日志记录添加级别标签，按配置为整条文本添加颜色。

        Args:
            record: logging 生成的日志记录对象。

        Returns:
            格式化后的日志字符串。
        """
        tag, color = TAGS.get(record.levelname, ("[info]", ""))
        record.tag = tag
        text = super().format(record)
        return f"{color}{text}{RESET}" if self.use_color else text


def setup_logging(level="INFO", log_file: Path | None = None) -> None:
    """配置根日志、控制台颜色及文件输出；重复调用避免重复添加处理器。

    Args:
        level: 根日志级别，如 INFO 或 DEBUG。
        log_file: 日志文件路径，默认写入项目 logs/service.log。

    Returns:
        None；可能创建日志目录和日志文件。
    """
    root = logging.getLogger()
    if getattr(root, "_tmf_configured", False) and log_file is None:
        return
    log_file = Path(log_file) if log_file is not None else LOG_DIR / "service.log"
    if getattr(root, "_tmf_configured", False):
        if log_file and not any(isinstance(handler, logging.FileHandler)
                                and Path(handler.baseFilename) == Path(log_file).resolve()
                                for handler in root.handlers):
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setFormatter(ColorFormatter(use_color=False))
            root.addHandler(handler)
        return

    root.setLevel(level)
    _enable_windows_ansi()
    color = os.environ.get("TMF_LOG_COLOR") != "0"

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColorFormatter(use_color=color))
    for handler in root.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
    root.addHandler(console)

    # 框架和第三方库已有的控制台 handler 也使用同一格式。
    for item in list(logging.root.manager.loggerDict.values()):
        if isinstance(item, logging.Logger):
            for handler in item.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    handler.setFormatter(ColorFormatter(use_color=color))
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "werkzeug"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    logging.captureWarnings(True)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(ColorFormatter(use_color=False))
        root.addHandler(file_handler)

    root._tmf_configured = True
    logging.getLogger(__name__).info("日志已初始化 level=%s color=%s file=%s", level, color, log_file)


def get_logger(name: str) -> logging.Logger:
    """确保日志系统已初始化，然后获取指定名称的 logger。

    Args:
        name: 日志记录器名称，通常传入 __name__。

    Returns:
        logging.Logger 实例。
    """
    setup_logging()
    return logging.getLogger(name)
