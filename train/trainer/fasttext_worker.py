"""独立进程运行原生 fastText，供父进程安全收集 C++ 标准错误进度。"""
import json
import sys
from pathlib import Path


if __name__ == "__main__":
    # debugpy 可能重复插入脚本目录；按路径移除所有项，避免同名模块遮蔽第三方包。
    worker_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != worker_dir]
    import fasttext

    model = fasttext.train_supervised(**json.loads(sys.argv[1]))
    model.save_model(sys.argv[2])
    names = {"epochs": "epoch", "lr": "lr", "dim": "dim", "word_ngrams": "wordNgrams",
             "bucket": "bucket", "minn": "minn", "maxn": "maxn"}
    Path(sys.argv[2]).with_name("training_config.json").write_text(
        json.dumps({key: getattr(model, value) for key, value in names.items()}), encoding="utf-8")
