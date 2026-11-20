# Redis 公共包

`from redis_service import redis_service` 使用共享实例，也可创建 `RedisService(client=..., url=...)`。

- `get_json(key)`：读取 JSON，键不存在返回 `None`。
- `set_json(key, value, ttl=None)`：写入 JSON，TTL 单位为秒。
- `start()`：检查连接，必要时启动本机 Redis。
- `stop()`：关闭本机 Redis（包括复用的实例）并释放连接。

连接默认使用 `redis://127.0.0.1:6379/0`，可通过 `TMF_REDIS_URL` 覆盖。
读写及 JSON 解码异常交给调用方处理，包内不依赖 inference 或数据库。
本机程序使用 `tools/redis/redis-server.exe` 或 PATH 中的 `redis-server`，日志位于 `runs/redis/redis.log`。
