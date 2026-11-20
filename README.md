# classifier_workshop

文本分类模型训练、模型管理与推理工作台，支持随机森林、fastText 和 BERT。

## 启动

```bash
pip install -r requirements.txt
python start.py
```

默认后端地址为 `http://127.0.0.1:8008`：

- `/training`：训练工作室。
- `/`：模型管理。
- `/inference`：模型推理。

项目包含 Windows Redis 程序，随后端启动、关闭。

## BERT 基座模型

基座模型不上传 Git，请自行下载 `bert-base-chinese`，放到项目根目录下的：

```text
models/
└── base_models/
    └── bert-base-chinese/
        ├── config.json
        ├── pytorch_model.bin
        ├── vocab.txt
        ├── tokenizer_config.json
        └── tokenizer.json
```

权重也可使用兼容的 `model.safetensors`。分词文件应与模型来自同一版本；
无需放入 TensorFlow 的 `tf_model.h5` 或 Flax 的 `flax_model.msgpack`。
此路径用于训练工作室的 BERT 训练。历史独立脚本可能使用各自配置中的路径。

## 本地文件

数据集、训练产物、模型权重、数据库、运行日志和编辑器配置均由 `.gitignore` 排除。
已发现含硬编码凭据的两个本地 LLM 脚本也被排除，不会上传。
LLM 调用所需的凭据应通过环境变量配置，不要提交 API Key。

## 验证

```bash
python -m unittest model_management.test_batch_delete inference.test_inference redis_service.test_service
node frontend/model_management/test_registry.cjs
node frontend/inference/test_chart.cjs
```

当前初始版本标记为 `v0`。
