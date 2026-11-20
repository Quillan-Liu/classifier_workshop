"""
模型推理模块
"""
import torch
from functools import lru_cache

try:
    from .bert_classifier_model import BertClassifierModel
    from .bert_config import config
except ImportError:
    from bert_classifier_model import BertClassifierModel
    from bert_config import config


@lru_cache(maxsize=1)
def load_model():
    model = BertClassifierModel().to(device=config.device)
    model.load_state_dict(torch.load(config.model_save_path, map_location=config.device, weights_only=True))
    model.eval()
    return model


def predict_many(texts):
    """返回与其他模型一致的 (类别编号, 类别名) 列表。"""
    model = load_model()
    results = []
    for start in range(0, len(texts), config.batch_size):
        inputs = config.tokenizer(
            texts[start:start + config.batch_size], return_tensors="pt",
            padding="max_length", truncation=True, max_length=config.pad_size,
        ).to(config.device)
        with torch.inference_mode():
            labels = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).argmax(dim=-1)
        results.extend((int(label), config.class_list[int(label)]) for label in labels)
    return results


def inference(data: dict):
    text = data.get('text', None)
    if not text or not text.strip():
        return {"text": text, "pred_class": None}
    return {"text": text, "pred_class": predict_many([text])[0][1]}


if __name__ == '__main__':
    data = {"text": "中华女子学院：本科层次仅1专业招男生"}
    result = inference(data)
    print('result--->\n', result)
