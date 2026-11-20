"""分层切分：同样的 seed 一定得到同样的划分。"""

from ..logging import log_call as _log_call

from collections import Counter

from sklearn.model_selection import train_test_split


def stratified_split(samples, ratio, seed):
    """按标签分层随机划分样本；类别不足或单类样本太少时退回普通随机划分。

    Args:
        samples: (文本, 标签) 二元组组成的样本序列。
        ratio: 从样本中切出的比例，必须在 0 与 1 之间。
        seed: 用于数据划分或训练的随机种子。

    Returns:
        二元组 (保留样本, 切出样本)；非法比例或无法切分时抛出 ValueError。

    Raises:
        ValueError: 比例不在 (0, 1)，或样本规模不足以完成划分。
    """

    with _log_call("train.dataset.splitter.stratified_split", locals()):
        if not 0 < ratio < 1:
            raise ValueError(f"切分比例必须在 (0, 1) 之间，当前为 {ratio}")

        labels = [label for _, label in samples]
        counts = Counter(labels)
        # 某类样本太少时 sklearn 的分层会直接报错，退回普通随机切分
        if len(counts) < 2 or min(counts.values()) < 2:
            return train_test_split(samples, test_size=ratio, random_state=seed)
        return train_test_split(samples, test_size=ratio, random_state=seed, stratify=labels)
