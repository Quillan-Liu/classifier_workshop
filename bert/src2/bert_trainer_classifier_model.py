import torch
import torch.nn as nn
from transformers import PreTrainedModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from .bert_trainer_config import config
from .bert_trainer_utils import build_dataloader


# 创建文本分类模型, 继承PreTrainedModel
class BertClassifierModel(PreTrainedModel):
    # 修改父类类属性
    config_class = BertConfig
    base_model_prefix = "bert"

    # 定义init方法
    def __init__(self, new_config=None):
        if new_config is None:
            new_config = config.bert_config
        super().__init__(new_config)

        # 定义bert模型结构
        self.bert = config.bert_model
        # 定义全连接层
        self.fc = nn.Linear(in_features=config.hidden_size, out_features=config.num_classes)
        # 定义类别数
        self.num_classes = config.num_classes

    # 定义forward方法
    def forward(self, input_ids, attention_mask, labels=None, return_dict=True):
        # 1.bert-base-chinese前向传播计算
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # print('output--->\n', output)
        # 2.全连接层前向传播计算
        logits = self.fc(output.pooler_output)
        # 3.返回结果
        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
        if return_dict:
            return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)
        else:
            return loss, logits


if __name__ == '__main__':
    model = BertClassifierModel().to(device='cpu')
    print(model)

    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    for batch in train_dataloader:
        # batch: dict  **batch->字典拆包
        result = model(**batch, return_dict=True)
        print(result)
        print(result.logits)
        exit()
