"""随机森林项目使用的路径和数据格式常量。"""

from pathlib import Path

if __package__ == "rf":
    from config import CLASS_PATH, ENCODING, PROJECT_DIR, RAW_DATA_DIR
else:
    from ..config import CLASS_PATH, ENCODING, PROJECT_DIR, RAW_DATA_DIR

RF_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_DIR = RF_DIR / "data"
MODEL_DIR = RF_DIR / "save_model"
RESULT_DIR = RF_DIR / "result"

TRAIN_PATH = RAW_DATA_DIR / "train.txt"
DEV_PATH = RAW_DATA_DIR / "dev.txt"
TEST_PATH = RAW_DATA_DIR / "test.txt"
STOPWORDS_PATH = RAW_DATA_DIR / "stopwords.txt"

PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "process_train.csv"
PROCESSED_DEV_PATH = PROCESSED_DATA_DIR / "process_dev.csv"
PROCESSED_TEST_PATH = PROCESSED_DATA_DIR / "process_test.csv"
VECTORIZER_PATH = PROCESSED_DATA_DIR / "tfidf_vectorizer.joblib"
VOCAB_PATH = PROCESSED_DATA_DIR / "vocabulary.tsv"

SPLITS = ("train", "dev", "test")
CSR_PATHS = {name: PROCESSED_DATA_DIR / f"{name}_tfidf.npz" for name in SPLITS}
LABEL_PATHS = {name: PROCESSED_DATA_DIR / f"{name}_labels.npy" for name in SPLITS}
RAW_DELIMITER = "\t"
OUTPUT_COLUMNS = ("text", "label", "words")

MODEL_PATH = MODEL_DIR / "tfidf_random_forest.joblib"
TRAIN_METRICS_PATH = RESULT_DIR / "train_metrics.json"
EVAL_METRICS_PATH = RESULT_DIR / "eval_metrics.json"
TFIDF_MAX_FEATURES = None  # 保留训练集词表中的所有非停用词
TRAIN_VALIDATION_SIZE = 0.2
RF_N_ESTIMATORS = 50
RF_TREES_PER_STEP = 10
RF_MAX_ROUNDS = 1
RF_EARLY_STOP_ACCURACY = 0.92
RF_RANDOM_STATE = 42
RF_N_JOBS = -1
