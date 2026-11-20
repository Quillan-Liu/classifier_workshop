"""
模型训练
一边训练一边验证
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score
from tqdm import tqdm
from .bert_config import config
from .bert_utils import build_dataloader
from .bert_classifier_model import BertClassifierModel
# 忽略的警告信息
import warnings

warnings.filterwarnings("ignore")


# 模型验证
def model2dev(model, dataloader):
    """
    模型验证
    :param model: 模型对象
    :param dataloader: 数据加载器对象
    :return: report f1_score acc precision
    """
    # 1.切换模型推理模式
    model.eval()
    # 2.定义空列表收集 预测标签 和 真实标签
    dev_preds, dev_labels = [], []
    # 3.循环遍历验证集数据加载器对象
    for input_ids, attention_mask, labels in dataloader:
        with torch.no_grad():
            # 4.模型推理
            y_pred = model(input_ids=input_ids, attention_mask=attention_mask)
            # print('y_pred--->\n', y_pred.shape, y_pred)
            # 5.将预测标签下标 和 真实标签下标保存到列表中
            dev_pred = torch.argmax(input=y_pred, dim=-1)
            dev_labels.extend(labels.tolist())
            dev_preds.extend(dev_pred.tolist())
    # 6.计算report f1_score acc precision
    report = classification_report(dev_labels, dev_preds)
    f1score = f1_score(dev_labels, dev_preds, average='micro')  # 使用微平均计算 F1 分数
    accuracy = accuracy_score(dev_labels, dev_preds)  # 计算准确度
    precision = precision_score(dev_labels, dev_preds, average='micro')  # 使用微平均计算精确度
    return report, f1score, accuracy, precision


# 训练函数
def model2train():
    # 1.构建数据加载器对象
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    # 2.实例化模型对象
    model = BertClassifierModel().to(device=config.device)
    # 3.实例化优化器对象
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    # 4.实例化损失函数对象
    criterion = nn.CrossEntropyLoss()
    # 5.模型训练
    # 初始化最佳F1-score分数 0分
    best_f1_score = 0.0
    # 5.1 外层循环遍历epoch
    for epoch in range(1, config.num_epochs + 1):
        total_loss = 0.0  # 每轮总损失值
        total_iters = 0  # 每轮总批次数
        # 预测标签 和 真实标签 列表
        train_preds, train_labels = [], []
        # 5.2 内层循环遍历batch 数据加载器对象
        for batch_idx, (input_ids, attention_mask, labels) in tqdm(enumerate(train_dataloader, start=1),
                                                                   desc=f"Bert Classifier Training Epoch {epoch}/{config.num_epochs}...."):
            # print('input_ids--->\n', input_ids.shape)
            # print('attention_mask--->\n', attention_mask.shape)
            # print('labels--->\n', labels.shape, labels)
            # 5.3 模型计算 前向传播
            # 5.4 损失计算
            # 切换模型训练模式
            model.train()
            y_pred, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # print('y_pred--->\n', y_pred.shape, y_pred)
            # 5.5 梯度清零
            optimizer.zero_grad()
            # 5.6 反向传播
            loss.backward()
            # 5.7 参数更新
            optimizer.step()
            # 5.8 统计训练过程中的信息 每轮总损失值 每轮总批次数 平均损失
            total_loss += loss.item()
            total_iters += 1
            avg_loss = total_loss / total_iters
            # 5.9 将预测标签下标 和 真实标签下标保存到列表中
            train_pred = torch.argmax(input=y_pred, dim=-1)
            # print('train_pred--->\n', train_pred.shape, train_pred)
            train_preds.extend(train_pred.tolist())
            train_labels.extend(labels.tolist())
            # 5.10 打印每100轮训练过程中的信息 并 进行模型验证 模型保存
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}/{config.num_epochs}")
                print(f"Train Loss: {avg_loss:.4f}")
                # 模型验证
                report, current_f1_score, accuracy, precision = model2dev(model, dev_dataloader)
                print(f"Dev Report: {report}")
                print(f"Dev F1 Score: {current_f1_score:.4f}")
                print(f"Dev Accuracy: {accuracy:.4f}")
                print(f"Dev Precision: {precision:.4f}")

                # 保存最佳模型
                if current_f1_score > best_f1_score:
                    # 更新最佳F1-score分数
                    best_f1_score = current_f1_score
                    # 保存模型参数
                    torch.save(model.state_dict(), config.model_save_path)
                    print(f"Best F1-score: {best_f1_score:.4f}")
        # 打印每轮分类评估报告
        report = classification_report(train_labels, train_preds, target_names=config.class_list, output_dict=True)
        print(f"Train Report: {report}")


if __name__ == '__main__':
    model2train()
