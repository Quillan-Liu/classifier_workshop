"""自定义网络结构，供训练、评估和推理共用。"""

from torch import nn


class BertClassifierModel(nn.Module):
    """BERT 池化输出接线性分类头，全参微调。"""

    def __init__(self, bert, num_classes):
        super().__init__()
        self.bert = bert
        self.fc = nn.Linear(bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(output.pooler_output)
        if labels is not None:
            return logits, nn.functional.cross_entropy(logits, labels)
        return logits
