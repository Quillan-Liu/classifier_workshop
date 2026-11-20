"""fastText 字符级新闻标题分类。"""

from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "save_models" / "fastText_char_default.bin"
CLASS_PATH = Path(__file__).resolve().parents[1] / "data" / "class.txt"


@lru_cache(maxsize=1)
def load_model():
    import fasttext

    return fasttext.load_model(str(MODEL_PATH))


@lru_cache(maxsize=1)
def class_names():
    return CLASS_PATH.read_text(encoding="utf-8").splitlines()


def predict_many(texts):
    """返回与 RF 推理相同的 (类别编号, 类别名) 列表。"""
    model = load_model()
    name_to_id = {name: index for index, name in enumerate(class_names())}
    results = []
    for text in texts:
        processed = " ".join(text.replace("：", ""))
        label = model.predict(processed)[0][0].removeprefix("__label__")
        results.append((name_to_id[label], label))
    return results


def inference(data: dict) -> dict:
    """兼容原有的单条推理调用。"""
    return {**data, "pred_class": predict_many([data["text"]])[0][1]}


if __name__ == '__main__':
    data = {'text': "中华女子学院：本科层次仅1专业招男生"}
    print(inference(data))
