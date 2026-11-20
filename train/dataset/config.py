"""数据集处理模块的默认配置。"""

from pathlib import Path

ENCODING = "utf-8"
PROJECT_DIR = Path(__file__).resolve().parents[2]

# 支持的用途
PURPOSES = ("bert", "fasttext", "rf")

# 默认切分比例（train 中切出）
DEFAULT_DEV_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_SEED = 42

# 原始数据的列名候选，按优先级排列
TEXT_COLUMNS = ("text", "sentence", "content", "query", "input")
LABEL_COLUMNS = ("label", "labels", "intent", "category", "target", "y")

# 项目自带的停用词表（仅 RF 的 TF-IDF 使用）
DEFAULT_STOPWORDS = PROJECT_DIR / "data" / "stopwords.txt"

# fastText 里要去掉的标点（保留语义，减少噪声 token）
FASTTEXT_PUNCT = "，。！？、：；“”‘’（）《》【】…—,.!?:;\"'()[]"
