"""
搭建文本分类网络模型
bert-base-chinese + 输出头

后续模型更新时会更新bert-base-chinese参数 以及 输出头参数 -> 全参微调
"""
import torch.nn as nn
import torch


# 创建模型类
class BertClassifierModel(nn.Module):
    def __init__(self, bert=None, num_classes=None):
        super().__init__()
        if bert is None:
            try:
                from .bert_config import config
            except ImportError:
                from bert_config import config
            bert, num_classes = config.bert_model, config.num_classes
        # 实例化bert-base-chinese模型结构
        self.bert = bert
        # 定义输出头结构
        # in_features: bert-base-chinese池化层的输出维度
        # out_features: 类别数 10分类问题
        self.fc = nn.Linear(in_features=bert.config.hidden_size, out_features=num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        # 1.bert-base-chinese前向传播计算
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # print('output--->\n', output)
        pooler_output = output.pooler_output
        # print('pooler_output--->\n', pooler_output)
        # print('pooler_output.shape--->\n', pooler_output.shape)

        # 2.bert-base-chinese结果给到输出头进行前向传播计算
        y_pred = self.fc(pooler_output)
        # print('y_pred--->\n', y_pred.shape, y_pred)

        if labels is not None:
            # 计算损失
            criterion = nn.CrossEntropyLoss()
            loss = criterion(y_pred, labels)
            return y_pred, loss
        return y_pred


if __name__ == '__main__':
    from .bert_config import config
    model = BertClassifierModel().to(device=config.device)
    print('model--->\n', model)
    # 修改bert-base-chinese的词嵌入层参数的梯度计算状态, 不进行参数更新
    # model.bert.embeddings.word_embeddings.weight.requires_grad_(False)
    # print(model.bert.embeddings.word_embeddings.weight.requires_grad)
    model.train()
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    for input_ids, attention_mask, labels in train_dataloader:
        y_pred = model(input_ids, attention_mask)
        break
