"""与模型无关的文本清洗。

只做三件事：全角空格归一、把制表符/换行压成普通空格、去掉首尾空白。
不去标点、不去停用词、不转小写 —— 这些属于各模型的派生阶段，
放在这里会让 BERT 和 RF 看到不一样的文本。
"""

import re

_WHITESPACE = re.compile(r"\s+")


def clean(text) -> str:
    """归一全角空格，将连续空白压成单个空格并去除首尾空白，不改大小写和标点。

    Args:
        text: 待清洗或分词的原始文本。

    Returns:
        清洗后的文本字符串；输入不是字符串时返回空字符串。
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("\u3000", " ")  # 全角空格
    text = text.replace("\t", " ")      # 制表符会破坏 text\tlabel 格式
    return _WHITESPACE.sub(" ", text).strip()
