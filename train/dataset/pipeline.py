"""主流程：读取 -> 清洗 -> 去重 -> 划分 -> 按用途导出。"""

from ..logging import log_call as _log_call, get_logger

import json
from collections import Counter
from pathlib import Path

from .config import (
    DEFAULT_DEV_RATIO, DEFAULT_SEED, DEFAULT_TEST_RATIO, ENCODING, PURPOSES,
)
from .exporters import EXPORTERS
from .normalize import clean
from .reader import load
from .splitter import stratified_split

DEFAULT_OUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "prepared"
logger = get_logger(__name__)


def _normalize_uses(use):
    """将单个或多个导出类型统一为列表，并展开 all。

    Args:
        use: 数据导出类型，可为单个类型、类型列表或 all。

    Returns:
        有效的导出类型列表；未知类型抛出 ValueError。

    Raises:
        ValueError: 传入未支持的导出类型。
    """
    with _log_call("train.dataset.pipeline._normalize_uses", locals()):
        if isinstance(use, str):
            uses = list(PURPOSES) if use == "all" else [use]
        else:
            uses = ["all" if u == "all" else u for u in use]
            uses = list(PURPOSES) if "all" in uses else uses
        unknown = [u for u in uses if u not in PURPOSES]
        if unknown:
            raise ValueError(f"不支持的用途 {unknown}，可选 {list(PURPOSES)} 或 'all'")
        return uses


def _build_label_map(groups, explicit_names, source=None):
    """将原始标签映射为连续整数，并从显式名称或类别文件中确定类名。

    Args:
        groups: 按分组组织的 (文本, 标签) 样本字典。
        explicit_names: 原始 label 到类别名的字典或类别名列表；None 时查找类别文件。
        source: 数据来源路径，用于查找相邻类别文件。

    Returns:
        二元组 (原始标签到编号的映射, 编号到类名的映射)。

    Raises:
        ValueError: 显式类别名数量与原始标签种类数不一致。
    """
    with _log_call("train.dataset.pipeline._build_label_map", locals()):
        unique = {str(label).strip() for samples in groups.values() for _, label in samples}
        numeric = all(v.lstrip("-").isdigit() for v in unique)
        ordered = sorted(unique, key=(lambda v: (int(v), v)) if numeric else None)
        mapping = {value: index for index, value in enumerate(ordered)}
        logger.info("类别识别 source=%s 数据原始标签=%r 数据类别数=%s 传入类别名称=%r 名称数量=%s",
                    source, ordered, len(ordered), explicit_names,
                    len(explicit_names) if explicit_names else "自动")

        if isinstance(explicit_names, dict):
            missing = set(ordered) - explicit_names.keys()
            extra = explicit_names.keys() - set(ordered)
            if missing or extra:
                raise ValueError(f"label 映射与数据不一致：缺少 {sorted(missing)}，多余 {sorted(extra)}")
            names = [explicit_names[label] for label in ordered]
        elif explicit_names:
            if len(explicit_names) != len(ordered):
                raise ValueError(
                    f"数据已识别出 {len(ordered)} 个标签: {ordered}；"
                    f"但 labels 类别名称参数只有 {len(explicit_names)} 项: {explicit_names!r}。"
                    "请留空以自动生成类别名称，或按英文逗号分隔填写等量类别名称。"
                )
            names = list(explicit_names)
        else:
            names = _find_class_file(source) or [f"label_{index}" for index in range(len(ordered))]
            if len(names) != len(ordered):
                names = [f"label_{index}" for index in range(len(ordered))]
        # 类别用 dict 表达：{编号: 类名}，下游不再依赖“列表下标即 id”的隐式约定
        classes = {index: name for index, name in enumerate(names)}
        return mapping, classes


def _find_class_file(source):
    """依次查找数据目录及父目录的类别文件，忽略空白行。

    Args:
        source: 数据来源路径，用于查找相邻类别文件。

    Returns:
        首个非空类别名称列表；没有找到时返回 None。
    """
    with _log_call("train.dataset.pipeline._find_class_file", locals()):
        if source is None:
            return None
        for candidate in (source / "class.txt", source / "labels.txt", source.parent / "class.txt"):
            if candidate.is_file():
                names = [line.strip() for line in candidate.read_text(encoding=ENCODING).splitlines()]
                names = [name for name in names if name]
                if names:
                    return names
        return None


def _clean(groups, mapping):
    """清洗文本、移除空文本，并将原始标签转换为统一编号。

    Args:
        groups: 按分组组织的 (文本, 标签) 样本字典。
        mapping: 原始标签字符串到连续整数编号的映射。

    Returns:
        清洗后的分组样本字典，不包含空分组。
    """
    with _log_call("train.dataset.pipeline._clean", locals()):
        cleaned = {}
        for split, samples in groups.items():
            rows = []
            for text, label in samples:
                text = clean(text)
                if not text:
                    continue
                rows.append((text, mapping[str(label).strip()]))
            if rows:
                cleaned[split] = rows
        return cleaned


SPLIT_PRIORITY = ("test", "dev", "train")  # 重复文本优先留在评估集，污染的训练样本被丢弃


def _dedupe(groups, allow_conflicts):
    """按 test、dev、train 优先级去重，避免评估文本泄漏到训练集。

    Args:
        groups: 按分组组织的 (文本, 标签) 样本字典。
        allow_conflicts: 是否容许同一文本有多个标签；容许时保留优先遇到的标签。

    Returns:
        二元组 (去重后的分组, 去重统计)；不允许的标签冲突抛出 ValueError。

    Raises:
        ValueError: 同一文本标签冲突，且未允许保留首次标签。
    """
    with _log_call("train.dataset.pipeline._dedupe", locals()):
        conflicts = []
        stats = {"within_split": 0, "cross_split": 0}
        owner = {}
        kept = {}
        ordered = [s for s in SPLIT_PRIORITY if s in groups]
        ordered += [s for s in groups if s not in SPLIT_PRIORITY]

        for split in ordered:
            rows = []
            for text, label in groups[split]:
                owner_split, owner_label = owner.get(text, (None, None))
                if owner_split is None:
                    owner[text] = (split, label)
                    rows.append((text, label))
                elif owner_label != label:
                    conflicts.append((text, owner_label, label))
                elif owner_split == split:
                    stats["within_split"] += 1
                else:
                    stats["cross_split"] += 1
            kept[split] = rows

        if conflicts and not allow_conflicts:
            preview = "\n".join(f"  {t!r}: {a} vs {b}" for t, a, b in conflicts[:10])
            raise ValueError(
                f"发现 {len(conflicts)} 条文本存在多个标签（同一句话被标成不同类），"
                f"前 10 条:\n{preview}\n确认无误可加 --allow-conflicts 保留首次出现的标签。"
            )
        return {split: rows for split, rows in kept.items() if rows}, stats


def _ensure_splits(groups, dev_ratio, test_ratio, seed):
    """保留已有分组，缺少验证集或测试集时从训练样本中切分。

    Args:
        groups: 按分组组织的 (文本, 标签) 样本字典。
        dev_ratio: 从训练样本中划出验证集的比例。
        test_ratio: 缺少独立测试集时划出测试集的比例。
        seed: 用于数据划分或训练的随机种子。

    Returns:
        包含 train、dev、test 三个样本列表的字典。
    """
    with _log_call("train.dataset.pipeline._ensure_splits", locals()):
        train = groups.get("train") or [row for rows in groups.values() for row in rows]
        dev = groups.get("dev")
        test = groups.get("test")
        if not dev:
            train, dev = stratified_split(list(train), dev_ratio, seed)
        if not test:
            train, test = stratified_split(list(train), test_ratio, seed)
        return {"train": list(train), "dev": list(dev), "test": list(test)}


def prepare(
    input_path,
    use=("bert",),
    out_dir=None,
    labels=None,
    dev_ratio=DEFAULT_DEV_RATIO,
    test_ratio=DEFAULT_TEST_RATIO,
    seed=DEFAULT_SEED,
    stopwords=None,
    allow_conflicts=False,
    verbose=True,
    test_input=None,
    word_level=False,
    tfidf_max_features=None,
):
    """读取、清洗、去重和划分数据，按训练类型导出文件并写入 manifest.json。

    Args:
        input_path: 原始训练数据文件或目录路径。
        use: 数据导出类型，可为单个类型、类型列表或 all。
        out_dir: 导出文件的目标目录；None 时按数据集名称生成默认目录。
        labels: 类别名称列表；接口表单中为逗号分隔字符串。
        dev_ratio: 从训练样本中划出验证集的比例。
        test_ratio: 缺少独立测试集时划出测试集的比例。
        seed: 用于数据划分或训练的随机种子。
        stopwords: RF 停用词文件路径；None 时尝试默认停用词文件。
        allow_conflicts: 是否容许同一文本有多个标签；容许时保留优先遇到的标签。
        verbose: 是否将数据准备摘要打印到终端。
        test_input: 独立测试数据文件或目录；None 时按比例划分测试集。
        word_level: fastText 是否使用词级分词；False 使用字符级。
        tfidf_max_features: RF 词表最大特征数；None 表示不限制。

    Returns:
        包含类别、划分数量、去重统计及产物信息的 manifest 字典。

    Raises:
        ValueError: 类别映射、标签冲突、测试类别或数据划分不符合要求。
    """
    with _log_call("train.dataset.pipeline.prepare", locals()):
        uses = _normalize_uses(use)
        source = Path(input_path)
        out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_ROOT / source.stem

        groups = load(source)
        if test_input is not None:
            test_rows = [row for rows in load(Path(test_input)).values() for row in rows]
            if not test_rows:
                raise ValueError(f"测试集为空: {test_input}")
            groups = {name: rows for name, rows in groups.items() if name != "test"}
            groups["test"] = test_rows

        mapping, classes = _build_label_map(groups, labels, source)
        cleaned = _clean(groups, mapping)

        if test_input is not None:
            train_labels = {label for _, label in cleaned.get("train", [])}
            unseen = {label for _, label in cleaned["test"]} - train_labels
            if unseen:
                raise ValueError(f"测试集里有训练集没出现过的类别: {sorted(unseen)}")

        cleaned, dedupe_stats = _dedupe(cleaned, allow_conflicts)
        splits = _ensure_splits(cleaned, dev_ratio, test_ratio, seed)

        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for purpose in uses:
            artifacts[purpose] = EXPORTERS[purpose](
                out_dir / purpose, splits, classes, stopwords=stopwords, seed=seed,
                word_level=word_level,
                tfidf_max_features=tfidf_max_features,
            )

        manifest = {
            "input": str(source.resolve()),
            "out_dir": str(out_dir.resolve()),
            "purposes": uses,
            "classes": classes,
            "label2id": {name: index for index, name in classes.items()},
            "seed": seed,
            "dev_ratio": dev_ratio,
            "test_ratio": test_ratio,
            "rows": {name: len(rows) for name, rows in splits.items()},
            "label_distribution": {
                name: dict(sorted(Counter(label for _, label in rows).items()))
                for name, rows in splits.items()
            },
            "dropped_duplicates": dedupe_stats,
            "artifacts": artifacts,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding=ENCODING
        )
        if verbose:
            print(f"数据集: {source}")
            print(f"类别({len(classes)}): {classes}")
            print(f"划分: {manifest['rows']}  去重: {dedupe_stats}")
            print(f"输出: {out_dir.resolve()}")
            for purpose, info in artifacts.items():
                print(f"  [{purpose}] {info}")
        return manifest
