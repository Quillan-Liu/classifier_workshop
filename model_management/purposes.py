"""用途英文标识用于命名，中文名称用于展示。"""

from enum import Enum

MODEL_TYPE_LABELS = {"rf": "随机森林", "fasttext": "fastText", "bert": "BERT"}

PURPOSE_EN_TO_ZH = {
    "news": "新闻分类",
    "intent": "意图识别",
    "sentiment": "情感分析",
    "topic": "主题分类",
    "spam": "垃圾信息识别",
    "risk": "风险识别",
    "content_safety": "内容安全审核",
    "question": "问题分类",
    "ticket": "工单分类",
    "complaint": "投诉分类",
    "language": "语种识别",
    "domain": "领域分类",
    "relevance": "相关性判断",
    "duplicate": "重复文本识别",
    "others": "其他",
}
PURPOSE_ZH_TO_EN = {label: code for code, label in PURPOSE_EN_TO_ZH.items()}
Purpose = Enum("Purpose", {code: code for code in PURPOSE_EN_TO_ZH}, type=str)
PURPOSE_OPTIONS = [{"value": code, "label": label} for code, label in PURPOSE_EN_TO_ZH.items()]


def purpose_code(value: str) -> str:
    """将用途枚举、英文标识或已知中文名称规范为命名用英文标识。

    Args:
        value: 待校验或转换的输入值。

    Returns:
        有效用途标识字符串；空值或未知用途抛出 ValueError。

    Raises:
        ValueError: 用途为空或不在预定义的中英文映射中。
    """
    if isinstance(value, Purpose):
        return value.value
    if not isinstance(value, str):
        raise ValueError("训练用途必填，请选择有效用途")
    value = value.strip()
    code = PURPOSE_ZH_TO_EN.get(value, value)
    if code not in PURPOSE_EN_TO_ZH:
        raise ValueError(f"不支持的训练用途: {value!r}，可选 {list(PURPOSE_EN_TO_ZH)}")
    return code
