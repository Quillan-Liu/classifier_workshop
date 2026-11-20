"""模型命名器：每个训练产物一个独一无二、可读、可反解的名字。

名字格式：

    {类型}_{日期}_{时间}_{用途}_{短序列}
    bert_20261119_203012_intent_7f3a2c
    fasttext_20261120_103000_intent_word_auto_7f3a2c

- 类型：rf / fasttext / bert
- 日期时间：训练开始时间（本地时区）
- 用途：训练任务的目的，如 intent / sentiment，只保留字母数字下划线
- 短序列：6 位十六进制，保证同一秒内多次训练也不重名
"""


import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from model_management.purposes import PURPOSE_ZH_TO_EN

MODELS = ("rf", "fasttext", "bert")
PATTERN = re.compile(
    r"^(?P<model>rf|fasttext|bert)_(?P<date>\d{8})_(?P<time>\d{6})_"
    r"(?P<purpose>.+)_(?P<seq>[0-9a-f]{6})$"
)
_UNSAFE = re.compile(r"[^0-9a-zA-Z]+")


def normalize_purpose(purpose: Optional[str]) -> str:
    """将已知中文用途映射为英文，再清理为最长 24 字符的命名片段。

    Args:
        purpose: 训练用途标识或已知中文用途；用于生成名称及保存用途。

    Returns:
        小写字母数字及下划线组成的用途标识；空结果使用 default。
    """
    purpose = PURPOSE_ZH_TO_EN.get(purpose, purpose)
    cleaned = _UNSAFE.sub("_", purpose or "").strip("_").lower()
    return (cleaned or "default")[:24]


@dataclass(frozen=True)
class ModelName:
    model: str
    when: datetime
    purpose: str
    seq: str
    tokenizer: Optional[str] = None
    training_mode: Optional[str] = None

    @property
    def text(self) -> str:
        """组合模型类型、日期、时间、用途和短序列，生成模型名称。

        Returns:
            完整模型名字符串。
        """
        mode = f"_{self.tokenizer}_{self.training_mode}" if self.tokenizer else ""
        return f"{self.model}_{self.when:%Y%m%d_%H%M%S}_{self.purpose}{mode}_{self.seq}"

    def __str__(self) -> str:
        """返回模型名称对象的可读字符串形式。

        Returns:
            与 text 属性相同的模型名。
        """
        return self.text

    @property
    def trained_at(self) -> str:
        """将模型命名时间格式化为本地日期时间。

        Returns:
            格式为 YYYY-MM-DD HH:MM:SS 的字符串。
        """
        return self.when.strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        """将模型名称各部分转为可序列化数据。

        Returns:
            包含 name、model、purpose、trained_at、seq 的字典。
        """
        return {
            "name": self.text,
            "model": self.model,
            "purpose": self.purpose,
            "trained_at": self.trained_at,
            "seq": self.seq,
        }


def make_model_name(model: str, purpose: Optional[str] = "default",
                    when: Optional[datetime] = None,
                    seq: Optional[str] = None, *, tokenizer: str = "char",
                    autotune: bool = False) -> ModelName:
    """校验模型类型并构造模型名称对象，不检查磁盘重名。

    Args:
        model: 模型类型标识，如 rf、fasttext 或 bert。
        purpose: 训练用途标识或已知中文用途；用于生成名称及保存用途。
        when: 模型名称使用的本地时间；None 时使用当前时间。
        seq: 模型名称的短序列；None 时生成随机六位十六进制序列。

    Returns:
        ModelName；不支持的模型类型抛出 ValueError。

    Raises:
        ValueError: 模型类型不受支持。
    """
    if model not in MODELS:
        raise ValueError(f"不支持的模型类型: {model!r}，可选 {list(MODELS)}")
    if model == "fasttext" and (tokenizer not in ("char", "word") or not isinstance(autotune, bool)):
        raise ValueError("fastText 命名需要 char/word 分词方式及布尔值 autotune")
    return ModelName(
        model=model,
        when=when or datetime.now(),
        purpose=normalize_purpose(purpose),
        seq=(seq or uuid.uuid4().hex[:6]).lower(),
        tokenizer=tokenizer if model == "fasttext" else None,
        training_mode=("auto" if autotune else "default") if model == "fasttext" else None,
    )


def new_model_name(model: str, purpose: Optional[str] = "default",
                   taken=None, *, tokenizer: str = "char", autotune: bool = False) -> ModelName:
    """生成名称并通过回调检查冲突，最多尝试八次。

    Args:
        model: 模型类型标识，如 rf、fasttext 或 bert。
        purpose: 训练用途标识或已知中文用途；用于生成名称及保存用途。
        taken: 检查名称是否已占用的回调，接收名称并返回布尔值。

    Returns:
        未冲突的 ModelName；连续冲突抛出 RuntimeError。

    Raises:
        RuntimeError: 连续八次生成名称均发生冲突。
    """
    taken = taken or (lambda name: False)
    for _ in range(8):
        candidate = make_model_name(model, purpose, tokenizer=tokenizer, autotune=autotune)
        if not taken(candidate.text):
            return candidate
    raise RuntimeError(f"连续 8 次生成的名字都冲突，模型 {model} 用途 {purpose}")


def parse_model_name(name: str) -> ModelName:
    """解析规范模型名中的类型、时间、用途及短序列。

    Args:
        name: 待解析的完整规范模型名。

    Returns:
        ModelName；名称或日期格式非法时抛出 ValueError。

    Raises:
        ValueError: 模型名格式或其中的日期时间无效。
    """
    match = PATTERN.match(name)
    if not match:
        raise ValueError(f"不是合法的模型名: {name!r}（应形如 bert_20261119_203012_intent_7f3a2c）")
    when = datetime.strptime(match["date"] + match["time"], "%Y%m%d%H%M%S")
    purpose = match["purpose"]
    tokenizer = training_mode = None
    if match["model"] == "fasttext":
        parts = purpose.rsplit("_", 2)
        if len(parts) == 3 and parts[1] in ("char", "word") and parts[2] in ("auto", "default"):
            purpose, tokenizer, training_mode = parts
    return ModelName(model=match["model"], when=when,
                     purpose=purpose, seq=match["seq"], tokenizer=tokenizer, training_mode=training_mode)


def is_model_name(name: str) -> bool:
    """检查字符串是否符合模型名称的正则格式。

    Args:
        name: 需要检查正则格式的模型名。

    Returns:
        是否匹配的布尔值；不校验日期是否真实存在。
    """
    return bool(PATTERN.match(name))


if __name__ == "__main__":
    seen = set()
    for _ in range(5):
        item = new_model_name("bert", "intent", taken=lambda n: n in seen)
        seen.add(item.text)
        print(f"{item.text}  |  训练时间 {item.trained_at}  |  用途 {item.purpose}")
    sample = parse_model_name(next(iter(seen)))
    print("反解:", sample.to_dict())
    print(f"5 次生成，去重后 {len(seen)} 个")
