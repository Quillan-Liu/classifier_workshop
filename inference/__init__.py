"""模型推理模块。

``api`` 提供 HTTP 路由，``cache`` 管理目录和标签缓存，``service`` 执行分类。
Redis 操作统一委托给 :mod:`redis_service`，模型权重不写入缓存。
"""
