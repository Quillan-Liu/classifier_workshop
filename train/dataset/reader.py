"""把各种格式的数据集文件读成统一的 {split: [(text, label)]}。

支持：
    - modelscope / HF 的 .arrow 文件（或包含 .arrow 的目录）
    - .jsonl（每行 {"text": ..., "label": ...}）
    - .txt / .tsv / .csv（每行 `text<TAB>label`）
    - HF DatasetDict 目录（含 state.json）
"""

from ..logging import log_call as _log_call

import json
from pathlib import Path

from .config import ENCODING, LABEL_COLUMNS, TEXT_COLUMNS

# 目录里的元数据/说明文件，不是样本，扫描时必须跳过
SKIP_NAMES = {
    "class.txt", "labels.txt", "vocabulary.tsv", "manifest.json",
    "dataset_info.json", "readme.md", "requirements.txt",
}


def _arrow_table(path):
    """先按 Arrow stream 读取，失败后按 Arrow file 格式读取。

    Args:
        path: 待读取的数据文件、目录或配置文件路径。

    Returns:
        完整 pyarrow Table。
    """
    with _log_call("train.dataset.reader._arrow_table", locals()):
        import pyarrow.ipc as ipc

        try:
            return ipc.open_stream(str(path)).read_all()
        except Exception:
            return ipc.open_file(str(path)).read_all()


def _pick_column(columns, candidates, what, path):
    """按候选顺序在列名中寻找第一个匹配项。

    Args:
        columns: 数据表中实际存在的列名集合。
        candidates: 按优先级排列的候选字段或列名。
        what: 字段的可读含义，用于缺失字段错误消息。
        path: 待读取的数据文件、目录或配置文件路径。

    Returns:
        匹配列名；没有匹配项时抛出带文件上下文的 KeyError。

    Raises:
        KeyError: 数据表没有任一候选列。
    """
    for name in candidates:
        if name in columns:
            return name
    raise KeyError(f"{path}: 找不到{what}列，实际列为 {list(columns)}")


def _split_of(name):
    """按文件名推断数据分组，test/eval 优先，其次 dev/valid。

    Args:
        name: 待处理的名称；具体含义见函数说明。

    Returns:
        test、dev 或默认的 train。
    """
    name = name.lower()
    if "test" in name or "eval" in name:
        return "test"
    if "dev" in name or "valid" in name:
        return "dev"
    return "train"


def _read_arrow(path):
    """读取 Arrow 表的文本列和标签列，组合为样本。

    Args:
        path: 待读取的数据文件、目录或配置文件路径。

    Returns:
        (文本, 标签) 二元组列表。
    """
    with _log_call("train.dataset.reader._read_arrow", locals()):
        table = _arrow_table(path)
        text_col = _pick_column(table.column_names, TEXT_COLUMNS, "文本", path)
        label_col = _pick_column(table.column_names, LABEL_COLUMNS, "标签", path)
        data = table.to_pydict()
        return list(zip(data[text_col], data[label_col]))


def _pick_key(obj, candidates, what, path):
    """按候选顺序读取对象中第一个存在字段的值。

    Args:
        obj: 从 JSON 行解析出的样本对象。
        candidates: 按优先级排列的候选字段或列名。
        what: 字段的可读含义，用于缺失字段错误消息。
        path: 待读取的数据文件、目录或配置文件路径。

    Returns:
        匹配字段的值；没有匹配项时抛出 KeyError。

    Raises:
        KeyError: 样本对象没有任一候选字段。
    """
    for name in candidates:
        if name in obj:
            return obj[name]
    raise KeyError(f"{path}: 找不到{what}字段，实际字段为 {list(obj)}")


def _read_jsonl(path):
    """逐行解析 JSON 文本及标签，忽略空行。

    Args:
        path: 待读取的数据文件、目录或配置文件路径。

    Returns:
        样本列表；JSON 非法时抛出包含文件和行号的 ValueError。

    Raises:
        ValueError: 非空行不是合法 JSON。
    """
    with _log_call("train.dataset.reader._read_jsonl", locals()):
        samples = []
        with open(path, encoding=ENCODING) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: JSON 解析失败 {exc}") from exc
                samples.append((
                    _pick_key(obj, TEXT_COLUMNS, "文本", path),
                    _pick_key(obj, LABEL_COLUMNS, "标签", path),
                ))
        return samples


def _read_text(path):
    """读取 text<TAB>label 格式的文本，以最后一个制表符分隔。

    Args:
        path: 待读取的数据文件、目录或配置文件路径。

    Returns:
        样本列表；缺少分隔符时抛出 ValueError。csv 后缀也按制表符读取。

    Raises:
        ValueError: 非空行缺少制表符。
    """
    with _log_call("train.dataset.reader._read_text", locals()):
        samples = []
        with open(path, encoding=ENCODING) as f:
            for lineno, line in enumerate(f, 1):
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                text, sep, label = line.rpartition("\t")
                if not sep:
                    raise ValueError(f"{path}:{lineno}: 缺少制表符，应为 `text<TAB>label`，实为 {line!r}")
                samples.append((text, label.strip()))
        return samples


def _read_hf_dataset_dict(path):
    """从磁盘加载 Hugging Face Dataset 或 DatasetDict 并提取 text/label。

    Args:
        path: 待读取的数据文件、目录或配置文件路径。

    Returns:
        {分组名称: 样本列表}，单数据集默认放入 train。
    """
    with _log_call("train.dataset.reader._read_hf_dataset_dict", locals()):
        from datasets import load_from_disk

        ds = load_from_disk(str(path))
        groups = {}
        if hasattr(ds, "items"):
            for split, part in ds.items():
                groups[split] = list(zip(part["text"], part["label"]))
        else:
            groups["train"] = list(zip(ds["text"], ds["label"]))
        return groups


def _read_file(path, split=None):
    """读取单个文件，并使用显式分组或按文件名推断分组。

    Args:
        path: 待读取的数据文件、目录或配置文件路径。
        split: 数据分组名，如 train、dev 或 test。

    Returns:
        二元组 (分组名称, 样本列表)。
    """
    with _log_call("train.dataset.reader._read_file", locals()):
        path = Path(path)
        return (split or _split_of(path.stem)), _read_one(path)


def _read_one(path):
    """按扩展名分派 Arrow、JSONL 或制表符文本读取器。

    Args:
        path: 待读取的数据文件、目录或配置文件路径。

    Returns:
        样本列表；不支持的扩展名抛出 ValueError。

    Raises:
        ValueError: 数据文件扩展名不受支持。
    """
    with _log_call("train.dataset.reader._read_one", locals()):
        suffix = path.suffix.lower()
        if suffix in (".arrow", ".feather"):
            return _read_arrow(path)
        if suffix in (".jsonl", ".json"):
            return _read_jsonl(path)
        if suffix in (".txt", ".tsv", ".csv"):
            return _read_text(path)
        raise ValueError(f"不支持的文件类型: {path}")


def _read_dir(path):
    """扫描目录并按分组汇总样本，优先读取 Arrow，其次磁盘数据集及其他文本。

    Args:
        path: 待读取的数据文件、目录或配置文件路径。

    Returns:
        {分组名称: 样本列表}；没有支持的数据文件时抛出 FileNotFoundError。

    Raises:
        FileNotFoundError: 目录中不存在支持的数据文件。
    """
    with _log_call("train.dataset.reader._read_dir", locals()):
        arrows = sorted(path.rglob("*.arrow"))
        if arrows:
            groups = {}
            for file in arrows:
                split, samples = _read_file(file)
                groups.setdefault(split, []).extend(samples)
            return groups
        if (path / "state.json").exists():
            return _read_hf_dataset_dict(path)
        others = sorted(
            p for p in path.rglob("*")
            if p.suffix.lower() in (".jsonl", ".json", ".txt", ".tsv", ".csv")
            and p.name.lower() not in SKIP_NAMES
        )
        if not others:
            raise FileNotFoundError(f"{path}: 目录下没有任何 .arrow/.jsonl/.txt 数据文件")
        groups = {}
        for file in others:
            split, samples = _read_file(file)
            groups.setdefault(split, []).extend(samples)
        return groups


def load(path):
    """从数据文件或目录读取并统一组织样本。

    Args:
        path: 待读取的数据文件、目录或配置文件路径。

    Returns:
        {分组名称: [(文本, 标签)]}；路径不存在或数据为空时抛出异常。

    Raises:
        FileNotFoundError: 输入路径不存在。
        ValueError: 数据为空、格式错误或文件类型不受支持。
    """
    with _log_call("train.dataset.reader.load", locals()):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"输入路径不存在: {path}")
        groups = _read_dir(path) if path.is_dir() else {_split_of(path.stem): _read_one(path)}
        if not groups or not any(groups.values()):
            raise ValueError(f"{path}: 没有读到任何样本")
        return groups
