"""随机森林：直接吃数据集处理产出的 TF-IDF 稀疏矩阵。"""

from ..logging import log_call as _log_call

from ..configs import get_config
from .base import BaseTrainer


class RandomForestTrainer(BaseTrainer):
    kind = "rf"
    required_files = (
        "train_tfidf.npz", "train_labels.npy",
        "test_tfidf.npz", "test_labels.npy",
    )

    def _metrics(self, y_true, y_pred):
        """计算准确率和宏平均 F1，保留四位小数。

        Args:
            y_true: 真实标签序列。
            y_pred: 模型预测标签序列，与真实标签一一对应。

        Returns:
            包含 accuracy 和 macro_f1 的字典。
        """
        from ..evaluator import scores
        return scores(y_true, y_pred)

    def fit(self) -> dict:
        """读取 TF-IDF 特征训练随机森林，保存模型并计算 test 和可用 dev 指标。

        Returns:
            以分组名为键的指标字典。
        """
        with _log_call("train.trainer.rf.RandomForestTrainer.fit", locals()):
            import joblib
            import numpy as np
            import scipy.sparse as sp
            from sklearn.ensemble import RandomForestClassifier

            data_dir, params = self.spec.data_dir, self.spec.params

            def load(name):
                """读取指定分组的 TF-IDF 稀疏矩阵和标签数组。

                Args:
                    name: 数据分组名，如 train、dev 或 test。

                Returns:
                    二元组 (特征矩阵, 标签数组)。
                """
                with _log_call("train.trainer.rf.RandomForestTrainer.fit.load", locals()):
                    return (sp.load_npz(data_dir / f"{name}_tfidf.npz"),
                            np.load(data_dir / f"{name}_labels.npy"))

            x_train, y_train = load("train")
            x_test, y_test = load("test")
            cfg = get_config("rf", params)
            self.logger.info("RF 输入 shape=%s 类别数=%s 配置=%s",
                             x_train.shape, len(set(y_train)), cfg.to_dict())

            clf = RandomForestClassifier(
                n_estimators=1,
                n_jobs=int(cfg.n_jobs),
                random_state=int(cfg.random_state),
                warm_start=True,
            )
            datasets = {"train": (x_train, y_train), "test": (x_test, y_test)}
            try:
                datasets["dev"] = load("dev")
            except FileNotFoundError:
                self.warn(f"没有 dev 特征文件，跳过 dev 评估: {data_dir}")
            if 'dev' not in datasets:
                raise ValueError('分轮训练和早停需要 dev 验证集，请重新处理数据集')
            total = cfg.n_estimators * cfg.max_rounds
            best_accuracy, best_trees, best_round = -1, 0, 0
            for round_index in range(cfg.max_rounds):
                start = round_index * cfg.n_estimators
                end = start + cfg.n_estimators
                # 末批不足 trees_per_step 时，仍在本轮边界完成训练和验证。
                count = start
                while count < end:
                    count = min(count + cfg.trees_per_step, end)
                    clf.set_params(n_estimators=count)
                    clf.fit(x_train, y_train)
                    batches_per_round = (cfg.n_estimators + cfg.trees_per_step - 1) // cfg.trees_per_step
                    iteration = round_index * batches_per_round + (count - start + cfg.trees_per_step - 1) // cfg.trees_per_step
                    self.report(round=round_index + 1, tree=count, total_trees=total,
                                iteration=iteration, total_iterations=batches_per_round * cfg.max_rounds)
                metrics = {split: self._metrics(y, clf.predict(x))
                           for split, (x, y) in datasets.items()}
                accuracy = float(np.mean(clf.predict(datasets['dev'][0]) == datasets['dev'][1]))
                if accuracy > best_accuracy:
                    best_accuracy, best_trees, best_round = accuracy, count, round_index + 1
                self.report(stage="epoch_evaluation", round=round_index + 1, tree=count, total_trees=total, **metrics)
                if metrics['dev']['accuracy'] >= cfg.early_stop_accuracy:
                    break
            clf.estimators_ = clf.estimators_[:best_trees]
            clf.n_estimators = best_trees
            metrics = {split: self._metrics(y, clf.predict(x))
                       for split, (x, y) in datasets.items()}
            metrics["selected_epoch"] = best_round
            self.report(tree=best_trees, total_trees=total, selected=True, **metrics)
            joblib.dump(clf, self.spec.model_dir / "random_forest.joblib")
            return metrics
