"""TMF 项目通用路径、编码和接口设置。"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_DIR / "data"
CLASS_PATH = RAW_DATA_DIR / "class.txt"
ENCODING = "utf-8"
API_HOST = "0.0.0.0"
API_PORT = 8005
