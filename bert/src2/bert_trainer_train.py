from sklearn.metrics import f1_score, accuracy_score, precision_score
from transformers import TrainingArguments, Trainer
from .bert_trainer_config import config
from .bert_trainer_utils import build_dataloader, collate_fn
from .bert_trainer_classifier_model import BertClassifierModel
import warnings
warnings.filterwarnings("ignore")


# 定义计算指标函数
def compute_metrics(eval_pred):
    """
    适用于 transformers Trainer 的评估函数，计算多分类任务下的 F1、准确率和精确率指标。

    参数说明：
        eval_pred (tuple): 元组(predictions, label_ids)
        predictions: 模型预测的logits（通常为二维数组: [batch_size, num_classes]）
        label_ids: 真实标签（[batch_size]）

    返回：
        dict: 评价指标，键为 'eval_f1', 'eval_accuracy', 'eval_precision'
    """
    # print('eval_pred--->\n', eval_pred)
    # predictions, labels 是 ndarray类型
    predictions, labels = eval_pred
    # 将logits转换为分类id
    predictions = predictions.argmax(axis=-1)
    # 微平均 F1
    f1 = f1_score(labels, predictions, average="micro")
    # 总体准确率
    accuracy = accuracy_score(labels, predictions)
    # 微平均精确率
    precision = precision_score(labels, predictions, average="micro")
    # 返回供Trainer自动记录的指标（eval_ 前缀不可变）
    return {"eval_f1": f1, "eval_accuracy": accuracy, "eval_precision": precision}

# 定义训练主流程
def model2train():
    """
    用于BERT文本分类模型的训练流程，基于transformers库的Trainer与TrainingArguments。

    步骤说明：
        1. 加载训练/验证/测试数据集
        2. 初始化BertClassifier模型
        3. 配置TrainingArguments
        4. 创建Trainer对象
        5. 启动训练
        6. 保存模型
        7. 在测试集上评估并打印主要指标

    参数：无，配置均源自 config.py

    返回：
        Trainer: 已训练完成的Trainer对象
    """
    # 1. 加载数据集（训练、验证、测试）
    train_dataset, dev_dataset, test_dataset = build_dataloader(dataset=True)

    # 2. 初始化BERT分类模型（全新权重或可由Trainer自动恢复断点/权重）
    model = BertClassifierModel()

    # 3. 配置训练参数
    # 兼容不同版本的transformers，'eval_strategy'（新）或 'evaluation_strategy'（旧）
    training_args = TrainingArguments(
        output_dir=config.output_dir,  # 输出目录
        num_train_epochs=config.num_epochs,  # 训练轮数
        per_device_train_batch_size=config.batch_size,  # 每卡/每设备训练batch
        per_device_eval_batch_size=config.batch_size,  # 每卡/每设备验证batch
        warmup_steps=config.warmup_steps,  # 学习率预热步数
        weight_decay=config.weight_decay,  # 权重衰减
        learning_rate=config.learning_rate,  # 学习率
        logging_steps=config.logging_steps,  # 日志打印间隔（步）
        eval_strategy="steps",  # 评估触发模式（新参数名）  旧参数名:valuation_strategy
        eval_steps=config.eval_steps,  # 评估间隔（步）
        save_strategy="steps",  # 保存模式
        save_steps=config.save_steps,  # 保存间隔（步）
        save_total_limit=config.save_total_limit,  # 最多保存几个模型
        load_best_model_at_end=True,  # 训练结束后自动恢复最优模型
        metric_for_best_model="eval_f1",  # 以何指标为“最优模型”
        greater_is_better=True)  # 指标越大越好

    # 4. 创建Trainer对象
    trainer = Trainer(
        model=model,  # BERT分类模型
        args=training_args,  # 训练/评估参数
        train_dataset=train_dataset,  # 训练集
        eval_dataset=dev_dataset,  # 验证集
        data_collator=collate_fn,  # 指定数据collate函数（处理batch的拼接与编码）
        compute_metrics=compute_metrics)  # 指标回调函数（字典）

    # 5. 启动训练；Trainer内部支持自动断点恢复、日志回调、模型保存
    print("开始训练...")
    trainer.train()

    # 6. 保存最终模型权重及配置，便于后续加载使用
    trainer.save_model(config.model_save_path)
    print(f"模型已保存到: {config.model_save_path}")

    # 7. 在测试集上做最终评估，输出全部主要指标
    print("在测试集上评估...")
    test_results = trainer.evaluate(test_dataset)
    print("测试集结果:")
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    # 主程序入口：调用训练主流程
    print("开始使用TrainingArguments和Trainer进行训练...")
    model2train()
    print("\n训练完成！")