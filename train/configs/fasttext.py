"""fastText 的训练配置。"""


from dataclasses import asdict, dataclass, fields
import re


@dataclass
class FastTextConfig:
    tokenizer: str = "char"   # char 字符级 / word jieba 词级
    autotune: bool = False
    autotune_duration: int = 300  # 自动搜索预算，单位秒
    thread: int = 1
    verbose: int = 3
    autotuneModelSize: str = "100M"
    epochs: int = 5
    lr: float = 0.1
    dim: int = 100
    word_ngrams: int = 1
    bucket: int = 2000000     # 与 ft/ft_char_default.py 使用的 fastText 默认值一致
    minn: int = 1
    maxn: int = 3

    def __post_init__(self):
        """校验 fastText 的分词方式必须为 char 或 word。

        Returns:
            None；非法分词方式抛出 ValueError。

        Raises:
            ValueError: 分词方式不是 char 或 word。
        """
        if self.tokenizer not in ("char", "word"):
            raise ValueError(f"tokenizer 只支持 char/word，收到 {self.tokenizer!r}")
        if not isinstance(self.autotune, bool):
            raise ValueError("autotune 必须为布尔值")
        if (isinstance(self.autotune_duration, bool) or not isinstance(self.autotune_duration, int)
                or self.autotune_duration <= 0):
            raise ValueError("autotune_duration 必须为正整数秒")
        for key, minimum in (("thread", 1), ("verbose", 0), ("minn", 0), ("maxn", 0)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} 必须为不小于 {minimum} 的整数")
        if self.maxn < self.minn:
            raise ValueError("maxn 不能小于 minn")
        if not isinstance(self.autotuneModelSize, str) or not re.fullmatch(r"[1-9][0-9]*[kKmMgG]?", self.autotuneModelSize):
            raise ValueError("autotuneModelSize 必须为正整数大小，如 100M")

    @classmethod
    def from_params(cls, params: dict | None = None) -> "FastTextConfig":
        """从已知且非 None 的参数构建配置，其他字段忽略，缺失字段采用默认值。

        Args:
            params: 训练参数字典；缺省字段使用模型默认配置。

        Returns:
            当前配置类的实例。
        """
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in (params or {}).items()
                      if key in known and value is not None})

    def to_dict(self) -> dict:
        """将数据类中的全部配置字段转换为字典。

        Returns:
            可用于接口输出或训练参数合并的配置字典。
        """
        return asdict(self)


if __name__ == "__main__":
    print(FastTextConfig.from_params({"epochs": 30, "word_ngrams": 3}).to_dict())
