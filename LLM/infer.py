"""将 LLM 新闻分类结果适配为 /predict 的返回格式。"""

from pathlib import Path

CLASSES = (Path(__file__).resolve().parents[1] / "data" / "class.txt").read_text(encoding="utf-8").splitlines()


def predict_many(texts):
    from .model2pred import model2pred

    results = []
    for text in texts:
        category = model2pred(text)[0]["category"]
        results.append((CLASSES.index(category), category))
    return results
