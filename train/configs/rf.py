"""随机森林的训练配置。"""


from dataclasses import asdict, dataclass, fields


@dataclass
class RFConfig:
    n_estimators: int = 50  # 每轮新增树数
    n_jobs: int = -1         # 与 rf/rf_config.py 一致，使用全部 CPU 核心
    random_state: int = 42
    tfidf_max_features: int | None = None
    train_validation_size: float = 0.2
    trees_per_step: int = 10
    max_rounds: int = 1
    early_stop_accuracy: float = 0.92

    def __post_init__(self):
        for key in ('n_estimators', 'trees_per_step', 'max_rounds'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} 必须是正整数')
        if self.tfidf_max_features is not None:
            if isinstance(self.tfidf_max_features, bool) or not isinstance(self.tfidf_max_features, int) or self.tfidf_max_features < 1:
                raise ValueError('tfidf_max_features 必须为正整数或 null（不限制）')
        if not isinstance(self.train_validation_size, (int, float)) or not 0 < self.train_validation_size < 1:
            raise ValueError('train_validation_size 必须在 0 和 1 之间')
        if not isinstance(self.early_stop_accuracy, (int, float)) or not 0 <= self.early_stop_accuracy <= 1:
            raise ValueError('early_stop_accuracy 必须在 0 和 1 之间')
        if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, int) or self.n_jobs == 0:
            raise ValueError('n_jobs 必须为非零整数')
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int) or not 0 <= self.random_state <= 4294967295:
            raise ValueError('random_state 必须为 0 到 4294967295 之间的整数')

    @classmethod
    def from_params(cls, params: dict | None = None) -> "RFConfig":
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
    print(RFConfig.from_params({"n_estimators": 50, "无关参数": 1}).to_dict())
