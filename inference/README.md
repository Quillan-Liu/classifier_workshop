# 推理模块

Redis 由独立的 `redis_service` 包管理，推理缓存通过其 `get_json` / `set_json` 接口读写；连接与进程生命周期见 `redis_service/README.md`。
后端调用 `catalog.start()` 完成 Redis 启动和目录预热，退出时调用 `catalog.stop()`。
查询模型列表与预测均通过 `catalog.get()` 读取缓存；未命中、数据格式异常或 Redis 不可用时读取 SQLite 并通过 `set_json(..., ttl=300)` 回填。
模型登记、修改和删除通过 `catalog.refresh()` 将最新目录和标签写入 Redis。

启动后端：`python start.py`，打开 `/inference`。独立前端：`node frontend/serve.mjs inference 3003`。
训练、模型管理、推理使用同一个后端。旧 Flask `app.py` 保留兼容。
`start.bat`、`python main.py` 和 `python app.py` 的旧 Flask 入口也会自动启动 Redis，并在服务正常退出、Ctrl+C 或运行异常时关闭 Redis。强制结束进程或断电无法执行退出清理。

安装 `pip install -r requirement.txt`。项目自带 `tools/redis/redis-server.exe`，后端启动时自动启动本机 Redis，已运行则复用，无需 Docker。默认 `127.0.0.1:6379`、数据库 `0`、无密码；PyCharm 使用相同参数连接。后端正常退出时同时关闭本机 Redis（包括启动时复用的本机实例），下次启动自动拉起。日志位于 `runs/redis/redis.log`，缓存不落盘，由 SQLite 重建。
服务启动预热模型信息和标签；请求先读 Redis，未命中读取 SQLite 并回填，Redis 不可用时回退数据库并记录日志。
模型登记、改名、删除同步刷新；缓存 TTL 为 300 秒。Redis 只保存元数据和标签，不保存模型权重。
标签持久化至 registry.sqlite3 的 labels 字段，兼容将旧 label_path 文件迁移入数据库。

GET /inference/models 返回用途、类型、模型列表及标签，按创建时间和 ID 倒序。
POST /inference/predict 接收 `{"purpose":"news","model_id":"登记ID","text":"输入字符串"}`，返回 label 和 class_name。
预测结果同时包含全部标签的 `probabilities`。前端将其提交给 `POST /inference/chart`，
请求格式为 `{"items":[{"class_name":"体育","probability":0.8}]}`，返回内存生成的 SVG。
图像不写入文件或 Redis，响应禁用缓存，直接展示到前端。
RF 需要 random_forest.joblib 和同次训练的 tfidf_vectorizer.joblib；旧模型需补齐向量器或重新训练。
fastText 使用 training_config.json 的分词配置；BERT 支持自定义 classifier.pt 和 Hugging Face 格式。
模型权重按请求加载，首次或大型模型推理可能较慢。

验证：`python -m unittest inference.test_inference`。
