"""Redis 公共接口：连接、JSON 读写及本机进程生命周期。"""
import json
import os

from redis import Redis

from .local import ensure_local_redis, stop_local_redis


class RedisService:
    """封装 Redis JSON 读写及本机服务生命周期。

    Args:
        client (redis.Redis, optional): 已创建的客户端，传入后忽略 ``url``。
        url (str, optional): Redis 连接 URL。默认读取 ``TMF_REDIS_URL``，
            未设置时使用 ``redis://127.0.0.1:6379/0``。

    Note:
        构造对象不建立连接，也不启动进程。默认客户端启用字符串解码，
        连接及读写超时均为 1 秒。读写异常交给调用方处理，不在此层回退数据库。

    Example::

        cache = RedisService()
        cache.start()
        try:
            cache.set_json("example:labels", {"0": "体育"}, ttl=300)
            labels = cache.get_json("example:labels")
        finally:
            cache.stop()
    """
    def __init__(self, client=None, url=None):
        self._client = client if client is not None else Redis.from_url(
            url or os.getenv("TMF_REDIS_URL", "redis://127.0.0.1:6379/0"),
            socket_connect_timeout=1, socket_timeout=1, decode_responses=True,
        )

    def get_json(self, key):
        """读取并解码指定键的 JSON 数据。

        Args:
            key (str): Redis 键名。

        Returns:
            object: 解码后的 JSON 值。键不存在或存储 JSON ``null`` 时返回
            ``None``，调用方可根据业务要求检查返回类型。

        Raises:
            redis.exceptions.RedisError: Redis 连接或读取失败。
            json.JSONDecodeError: 缓存内容不是有效 JSON。
        """
        value = self._client.get(key)
        return json.loads(value) if value is not None else None

    def set_json(self, key, value, ttl=None):
        """将数据编码为 JSON 并覆盖指定键，保留中文字符。

        Args:
            key (str): Redis 键名。
            value (object): 可被 JSON 序列化的数据。
            ttl (int, optional): 有效秒数，须为正整数；默认不设置过期时间。

        Returns:
            bool: Redis 客户端返回的写入结果，成功时为 ``True``。

        Raises:
            TypeError: 数据无法被 JSON 序列化。
            ValueError: 数据包含循环引用。
            redis.exceptions.RedisError: 写入失败或过期时间不合法。
        """
        return self._client.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)

    def start(self):
        """检查 Redis 连接，必要时启动本机实例。

        Raises:
            RuntimeError: 连接不可用且地址非本机、未找到程序或启动失败。

        Note:
            已可连接的实例直接复用；本机进程的启动细节见
            :func:`redis_service.local.ensure_local_redis`。
        """
        ensure_local_redis(self._client)

    def stop(self):
        """关闭本机 Redis 并释放连接，不保存缓存快照。

        Warning:
            本机会关闭该地址的整个实例，包括启动时复用的实例，而非仅断开
            当前客户端。非本机地址直接返回，不执行关闭或连接释放。
        """
        stop_local_redis(self._client)


redis_service = RedisService()
