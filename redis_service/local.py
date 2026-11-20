"""本机 Redis 随后端启动；已运行的实例直接复用。"""
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time

from redis.exceptions import ConnectionError, TimeoutError

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def ensure_local_redis(client):
    """复用可用实例，或启动本机 Redis 并等待就绪。

    Args:
        client (redis.Redis): 用于探测实例及读取连接参数的客户端。

    Raises:
        RuntimeError: 不可达实例位于远程主机、未找到 Redis 程序，
            或进程未能在重试次数内就绪。
        OSError: 无法创建日志目录、打开日志文件或启动进程。

    Note:
        优先使用项目自带程序，其次查找 PATH。最多探测 30 次，每次失败后
        等待 0.1 秒；总耗时还包含客户端连接超时。进程绑定 IPv4 回环地址，
        关闭 RDB 和 AOF 持久化，日志写入 ``runs/redis/redis.log``。
    """
    try:
        if client.ping():
            logger.info("Redis 已就绪")
            return
    except (ConnectionError, TimeoutError):
        pass
    settings = client.connection_pool.connection_kwargs
    if settings.get("host", "localhost") not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("本项目仅自动启动本机 Redis，请使用本机地址")
    executable = ROOT / "tools" / "redis" / "redis-server.exe"
    command = str(executable) if executable.is_file() else shutil.which("redis-server")
    if not command:
        raise RuntimeError("未找到本机 Redis 程序：tools/redis/redis-server.exe")
    directory = ROOT / "runs" / "redis"
    directory.mkdir(parents=True, exist_ok=True)
    arguments = [command, "--bind", "127.0.0.1", "--port", str(settings.get("port", 6379)),
                 "--protected-mode", "yes", "--dir", str(directory), "--save", "", "--appendonly", "no"]
    if settings.get("password"):
        arguments.extend(["--requirepass", settings["password"]])
    # 缓存可从 SQLite 重建；日志写入项目目录，Redis 随后端生命周期关闭。
    with (directory / "redis.log").open("ab") as output:
        process = subprocess.Popen(arguments, cwd=directory, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    for _ in range(30):
        try:
            if client.ping():
                logger.info("本机 Redis 已启动，端口 %s", settings.get("port", 6379))
                return
        except (ConnectionError, TimeoutError):
            pass
        if process.poll() is not None:
            break
        time.sleep(0.1)
    raise RuntimeError(f"本机 Redis 启动失败，请查看 {directory / 'redis.log'}")


def stop_local_redis(client):
    """关闭本机实例并释放客户端连接，缓存由下次启动重建。

    Args:
        client (redis.Redis): 指向待关闭实例的客户端。

    Note:
        仅处理 ``localhost``、``127.0.0.1`` 和 ``::1``。发送
        ``SHUTDOWN NOSAVE`` 后释放连接；连接不可用或超时时记录警告并
        继续释放连接，其他异常仍向上传播。非本机地址不执行任何操作。
    """
    settings = client.connection_pool.connection_kwargs
    if settings.get("host", "localhost") not in {"127.0.0.1", "localhost", "::1"}:
        return
    try:
        client.shutdown(nosave=True)
        logger.info("本机 Redis 已关闭")
    except (ConnectionError, TimeoutError) as exc:
        logger.warning("关闭本机 Redis 时连接不可用: %s", exc)
    finally:
        client.close()
