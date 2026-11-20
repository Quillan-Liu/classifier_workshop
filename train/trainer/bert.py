"""沿用 bert.src 的分类模型与 PyTorch 训练循环。"""

from ..logging import log_call as _log_call

import json
import math
from functools import partial

from ..configs import get_config
from .base import BaseTrainer


class BertTrainer(BaseTrainer):
    kind = "bert"
    required_files = ("train.txt", "dev.txt", "test.txt", "class.txt")

    @staticmethod
    def _collate(batch, *, tokenizer, max_length, device):
        """将一批文本分词、截断、补齐，并将输入和标签移到训练设备。"""
        import torch

        texts, labels = zip(*batch)
        inputs = tokenizer(list(texts), padding="max_length", truncation=True,
                           max_length=max_length, return_tensors="pt")
        return (inputs["input_ids"].to(device), inputs["attention_mask"].to(device),
                torch.tensor(labels, dtype=torch.long, device=device))

    def _train_epoch(self, model, loader, optimizer, scheduler,
                    epoch, step, total_steps):
        """在训练模式下完成一轮梯度更新，返回累计训练步数。"""
        import torch

        model.train()
        total_loss = 0.0
        with torch.enable_grad():
            for batch, (ids, mask, labels) in enumerate(loader, start=1):
                if step >= total_steps:
                    break
                optimizer.zero_grad()
                _, loss = model(ids, mask, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                total_loss += loss.item()
                self.report(epoch=epoch, batch=batch, step=step, total_steps=total_steps,
                            batch_loss=loss.item(), average_batch_loss=total_loss / batch,
                            learning_rate=optimizer.param_groups[0]["lr"])
        return step

    def _evaluate(self, model, loader):
        """无梯度评估；无论成功或异常，都恢复调用前的训练/评估模式。"""
        import torch
        from ..evaluator import scores

        was_training = model.training
        model.eval()
        try:
            actual, predicted = [], []
            loss_sum, count = 0.0, 0
            with torch.inference_mode():
                for ids, mask, labels in loader:
                    logits, loss = model(ids, mask, labels)
                    loss_sum += loss.item() * len(labels)
                    count += len(labels)
                    actual.extend(labels.cpu().tolist())
                    predicted.extend(logits.argmax(-1).cpu().tolist())
            if count == 0:
                raise ValueError("评估数据不能为空")
            return {**scores(actual, predicted), "loss": loss_sum / count}
        finally:
            model.train(was_training)

    def fit(self):
        """微调 BERT 分类器，保存验证集 F1 最优权重、分词器和分类配置。

        Returns:
            dev/test 指标字典；配置或数据非法时抛出 ValueError。

        Raises:
            ValueError: 训练参数非正数、样本为空或标签编号越界。
        """
        with _log_call("train.trainer.bert.BertTrainer.fit", locals()):
            import torch
            from torch.utils.data import DataLoader
            from transformers import BertModel, BertTokenizer
            from bert.src.bert_classifier_model import BertClassifierModel
            from ..evaluator import _read_tsv

            cfg = get_config("bert", self.spec.params)
            if cfg.epochs <= 0 or min(cfg.batch_size, cfg.eval_batch_size, cfg.max_length) <= 0:
                raise ValueError("epochs、batch_size、eval_batch_size、max_length 必须大于 0")
            torch.manual_seed(int(cfg.seed))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = BertTokenizer.from_pretrained(self.spec.base_model)
            names = self.spec.data_dir.joinpath("class.txt").read_text(encoding="utf-8").splitlines()
            model = BertClassifierModel(BertModel.from_pretrained(self.spec.base_model), len(names)).to(device)

            collate_fn = partial(self._collate, tokenizer=tokenizer,
                                 max_length=int(cfg.max_length), device=device)

            loaders = {}
            for split in ("train", "dev", "test"):
                texts, labels = _read_tsv(self.spec.data_dir / f"{split}.txt")
                if not texts or any(label < 0 or label >= len(names) for label in labels):
                    raise ValueError(f"{split}: 数据为空或标签编号超出 class.txt 范围")
                loaders[split] = DataLoader(list(zip(texts, labels)), shuffle=split == "train",
                    collate_fn=collate_fn, batch_size=int(cfg.batch_size if split == "train" else cfg.eval_batch_size))

            optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
            total_steps = math.ceil(float(cfg.epochs) * len(loaders["train"]))
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: min(1.0, (step + 1) / max(1, int(cfg.warmup_steps))))

            output = self.spec.model_dir / "final"
            output.mkdir(parents=True, exist_ok=True)
            best, step, best_epoch = -1.0, 0, 0
            self.logger.info("BERT PyTorch FP32 训练 device=%s 配置=%s", device, cfg.to_dict())
            for epoch in range(math.ceil(float(cfg.epochs))):
                step = self._train_epoch(model, loaders["train"], optimizer, scheduler,
                                        epoch + 1, step, total_steps)
                metrics = {split: self._evaluate(model, loaders[split]) for split in ("train", "dev", "test")}
                self.report(stage="epoch_evaluation", epoch=epoch + 1, step=step, **metrics)
                if metrics["dev"]["f1"] > best:
                    best = metrics["dev"]["f1"]
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), output / "classifier.pt")
                if step >= total_steps:
                    break

            model.load_state_dict(torch.load(output / "classifier.pt", map_location=device, weights_only=True))
            model.bert.config.save_pretrained(output)
            tokenizer.save_pretrained(output)
            (output / "classifier.json").write_text(json.dumps({
                "class_names": names, "max_length": int(cfg.max_length),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            metrics = {split: self._evaluate(model, loaders[split]) for split in ("train", "dev", "test")}
            metrics["selected_epoch"] = best_epoch
            self.report(stage="best_checkpoint", **metrics)
            return metrics
