# 项目日志

所有运行、训练和临时检查的 `.log` 文件统一放在此目录。

- `service.log`：统一日志模块的默认输出，路径固定在项目中，不受启动目录影响。
- `train/模型名/模型名_log.log`：按完整模型名保存训练过程日志，包含数据处理、各轮评估、原生 fastText 进度和异常；并发任务按线程隔离。命令行训练使用模型输出目录名作为模型名。
- `train/模型名/training_history.jsonl`：同一模型的结构化进度和评估记录。
- `*_check.log` 等：历史检查日志；以后手动重定向输出也请写到此目录，例如 `python start.py --no-serve > logs/openapi_export.log 2>&1`。

代码请使用 `train.logging.get_logger()` 或 `setup_logging()`；需要单独文件时，将 `log_file` 指向此目录下的文件。
