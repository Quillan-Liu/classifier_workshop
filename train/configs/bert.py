"""BERT 微调配置：训练规模默认与 bert/src/bert_config.py 一致，使用 FP32。"""


from dataclasses import asdict, dataclass, fields


@dataclass
class BertConfig:
    epochs: float = 2
    batch_size: int = 8
    eval_batch_size: int = 8
    lr: float = 5e-5
    max_length: int = 32
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 50
    seed: int = 42

    @classmethod
    def from_params(cls, params: dict | None = None) -> "BertConfig":
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
    print(BertConfig.from_params({"epochs": 5, "batch_size": 8}).to_dict())
