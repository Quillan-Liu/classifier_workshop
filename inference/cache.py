"""模型目录及标签：Redis 优先，未命中读取 SQLite 并回填。"""
import logging
from model_management.store import ModelStore
from redis_service import redis_service

logger = logging.getLogger(__name__)

class Catalog:
    """管理推理使用的模型目录及标签映射。

    Args:
        store (ModelStore, optional): 元数据来源，默认使用项目模型数据库。
        cache (RedisService, optional): 提供 JSON 读写和生命周期接口的缓存服务。
            默认使用共享 Redis 服务，也可注入兼容接口的对象。

    Note:
        缓存整个目录，有效期为 300 秒。Redis 读写失败时使用 SQLite；
        数据库异常继续向上传播，避免将读取失败误判为空目录。
    """
    key = "tmf:inference:catalog:v1"

    def __init__(self, store=None, cache=None):
        self.store = store or ModelStore()
        self.cache = cache if cache is not None else redis_service

    def start(self):
        """启动 Redis，并从数据库预热目录和标签。

        Note:
            预热期间若数据库读取失败，会先关闭 Redis 再抛出原异常。
            Redis 写入失败由 :meth:`refresh` 记录日志，不阻止服务启动。
        """
        self.cache.start()
        try:
            self.refresh()
        except Exception:
            self.cache.stop()
            raise

    def stop(self):
        """通过 Redis 包关闭缓存服务。"""
        self.cache.stop()

    def refresh(self):
        """从数据库重建目录，并通过 Redis 接口覆盖缓存。

        Returns:
            list[dict]: 数据库中的有效模型记录，包含路径、参数和标签映射。
                Redis 写入失败时仍返回数据库结果。
        """
        items = self.store.inference_catalog()
        try:
            self.cache.set_json(self.key, items, ttl=300)
        except Exception as exc:
            logger.warning("Redis 缓存写入失败，使用数据库: %s", exc)
        return items

    def get(self):
        """优先读取缓存，未命中或读取失败时刷新目录。

        Returns:
            list[dict]: 缓存或数据库中的模型目录；空列表也是有效缓存。

        Note:
            仅接受列表形式的缓存。JSON 解码失败或顶层类型不符时，
            使用数据库重建缓存，不向调用方返回损坏的数据。
        """
        try:
            items = self.cache.get_json(self.key)
            if isinstance(items, list):
                return items
        except Exception as exc:
            logger.warning("Redis 缓存读取失败，使用数据库: %s", exc)
        return self.refresh()

catalog = Catalog()
