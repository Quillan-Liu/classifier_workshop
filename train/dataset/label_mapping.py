"""读取训练用的原始 label 到类别名称映射。"""

import json
from pathlib import Path


def load_label_mapping(path):
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("映射文件必须使用 UTF-8 编码") from exc
    pairs = []
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text, object_pairs_hook=lambda items: pairs.extend(items) or dict(items))
        except json.JSONDecodeError as exc:
            raise ValueError(f"映射文件不是合法 JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError('映射 JSON 必须为对象，例如 {"0":"体育","1":"财经"}')
    elif path.suffix.lower() in (".txt", ".tsv"):
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"映射文件第 {number} 行应为 label<TAB>类别名")
            pairs.append(parts)
    else:
        raise ValueError("映射文件仅支持 JSON、TXT 或 TSV")
    mapping = {}
    for label, name in pairs:
        if not isinstance(label, str) or not isinstance(name, str) or not label.strip() or not name.strip():
            raise ValueError("映射的 label 和类别名必须是非空字符串")
        label, name = label.strip(), name.strip()
        if label in mapping:
            raise ValueError(f"映射文件存在重复 label: {label}")
        mapping[label] = name
    if not mapping or len(set(mapping.values())) != len(mapping):
        raise ValueError("映射文件不能为空，类别名称不能重复")
    return mapping
