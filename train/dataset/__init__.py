"""数据集处理模块：输入数据集 + 用途，输出可直接训练的产物。

    from .dataset import prepare

    prepare("D:/datasets/xxx", use="bert", out_dir="D:/tmf/data/intent", labels=[...])
    prepare("D:/datasets/xxx", use="all",  out_dir="D:/tmf/data/intent")
"""

from .pipeline import prepare

__all__ = ["prepare"]
