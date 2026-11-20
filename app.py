"""Flask 推理接口：python app.py 或 python -m tmf.app。"""

import asyncio

if __package__:
    from .train.logging import setup_logging
else:
    from train.logging import setup_logging

from flask import Flask, jsonify, request, send_from_directory

if __package__:
    from .config import API_HOST, API_PORT, PROJECT_DIR
    from .ft.infer import predict_many as ft_predict_many
    from .LLM.infer import predict_many as llm_predict_many
    from .rf.rf_infer import predict_many as rf_predict_many
else:
    from config import API_HOST, API_PORT, PROJECT_DIR
    from ft.infer import predict_many as ft_predict_many
    from LLM.infer import predict_many as llm_predict_many
    from rf.rf_infer import predict_many as rf_predict_many


def bert_predict_many(texts):
    if __package__:
        from .bert.src.bert_infer import predict_many
    else:
        from bert.src.bert_infer import predict_many
    return predict_many(texts)


def bert_trainer_predict_many(texts):
    if __package__:
        from .bert.src2.bert_trainer_inference import predict
        from .bert.src2.bert_trainer_config import config
    else:
        from bert.src2.bert_trainer_inference import predict
        from bert.src2.bert_trainer_config import config
    return [(config.class_list.index(name), name) for name in (predict({"text": text})["pred_class"] for text in texts)]


MODELS = {"rf": rf_predict_many, "ft": ft_predict_many, "bert": bert_predict_many, "bert-trainer": bert_trainer_predict_many, "llm": llm_predict_many}
app = Flask(__name__, static_folder=str(PROJECT_DIR / "frontend"), static_url_path="/static")
setup_logging()


@app.get("/")
def index():
    return send_from_directory(PROJECT_DIR / "frontend", "index.html")


@app.post("/predict")
async def predict():
    """每个请求独立选择模型，预测单条 text 或批量 texts。"""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="请提交 JSON 对象"), 400

    model = data.get("model", "rf")
    if not isinstance(model, str) or model not in MODELS:
        return jsonify(error=f"不支持的模型：{model}"), 400

    single = "text" in data
    texts = [data["text"]] if single else data.get("texts")
    if not isinstance(texts, list) or not texts or any(
        not isinstance(text, str) or not text.strip() for text in texts
    ):
        return jsonify(error="text 必须是非空字符串，或 texts 为非空字符串列表"), 400

    predictions = await asyncio.to_thread(MODELS[model], texts)
    results = [
        {"label": label, "class_name": name}
        for label, name in predictions
    ]
    return jsonify(results[0] if single else {"results": results})


def run_server():
    """启动网页服务，并在服务退出时关闭 Redis。"""
    if __package__:
        from .redis_service import redis_service
    else:
        from redis_service import redis_service
    redis_service.start()
    try:
        app.run(host=API_HOST, port=API_PORT, threaded=True)
    finally:
        redis_service.stop()


if __name__ == "__main__":
    run_server()
