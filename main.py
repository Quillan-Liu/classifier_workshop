"""启动新闻标题分类网页和推理接口。"""

from app import run_server
from config import API_HOST, API_PORT
from train.logging import get_logger

logger = get_logger(__name__)


def main():
    logger.info("网页：http://127.0.0.1:%s/", API_PORT)
    logger.info("接口：http://127.0.0.1:%s/predict", API_PORT)
    run_server()


if __name__ == "__main__":
    main()
