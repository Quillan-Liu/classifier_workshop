"""
学生模型 BiLSTM
也可以使用TextCNN作为学生模型
"""
import torch
import torch.nn as nn
from .config import config
from bert.src.bert_utils import build_dataloader


class BiLSTMClassifier(nn.Module):
    """
    BiLSTM + 全连接层的分类模型，作为学生模型。
    """

    def __init__(self,
                 embed_size=config.embed_size,
                 hidden_size=config.hidden_size_lstm,
                 num_layers=config.num_layers,
                 num_classes=config.num_classes,
                 dropout=config.dropout):
        """
        初始化BiLSTM模型。
        参数：
            embed_size: 嵌入维度。
            hidden_size: LSTM隐藏状态维度。
            num_layers: LSTM层数。
            num_classes: 分类类别数。
            dropout: Dropout比例。
        """
        super(BiLSTMClassifier, self).__init__()
        self.vocab_size = config.tokenizer.vocab_size  # 从BERT分词器动态获取词汇表大小
        self.hidden_size = hidden_size

        # 定义词嵌入层
        self.embedding = nn.Embedding(self.vocab_size, embed_size, padding_idx=0)

        # 定义LSTM层 隐层
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=self.hidden_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True,  # (句子数, 词数, 特征数)
                            dropout=dropout)

        # 定义全连接层
        # hidden_size * 2: 双向, 双倍特征
        self.fc = nn.Linear(in_features=self.hidden_size * 2, out_features=num_classes)

        # 定义dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        """
        前向传播，仅在嵌入层使用 attention_mask 进行掩码处理。
        参数：
            input_ids: 输入的token ID，形状为 [batch_size, seq_len]。
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len]，1 表示有效 token，0 表示填充 token。
        返回：
            logits: 分类logits，形状为 [batch_size, num_classes]。
        """
        # print('input_ids--->\n', input_ids.shape)
        # 嵌入层
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_size]
        # print('embedded--->\n', embedded.shape)

        # LSTM 层
        # lstm_output: 最后一层LSTM层每个时间步的输出隐藏状态值集合
        lstm_output, (hidden, cell) = self.lstm(embedded)  # lstm_out: [batch_size, seq_len, hidden_size*2]
        # print('lstm_output--->\n', lstm_output.shape)

        # 获取最后一个词的语义表示
        # 由于 input_ids 包含填充词 (0)，直接使用 [:, -1, :] 可能获取到填充位置的输出
        # 需要根据 attention_mask 找到每个序列中最后一个非 padding 的位置
        # attention_mask_sum: [batch, ] 表示每个序列的有效长度
        # print('attention_mask--->\n', attention_mask.shape, attention_mask)
        lengths = attention_mask.sum(dim=1).long()  # shape: [batch_size]
        # print('lengths--->\n', lengths.shape, lengths)
        # 构造索引，获取每个样本有效长度的最后一个时间步的隐藏状态
        # 注意：lengths - 1 是因为索引从 0 开始
        last_indices = lengths - 1
        # print('last_indices--->\n', last_indices.shape, last_indices)

        # 提取 [batch, last_indices, :]
        # 创建一个 [0, 1, ..., batch_size-1] 的张量作为第一维索引
        batch_size = lstm_output.size(0)
        batch_indices = torch.arange(batch_size, device=config.device)
        # print('batch_indices--->\n', batch_indices.shape, batch_indices)
        # 提取关键特征：形状 [batch, hidden_size * 2]
        # 正向真实最后一个时间步的输出隐藏状态
        f_last_hidden_state = lstm_output[batch_indices, last_indices, :self.hidden_size]
        # print('f_last_hidden_state--->\n', f_last_hidden_state.shape)
        # 反向真实最后一个时间步的输出隐藏状态
        b_last_hidden_state = lstm_output[batch_indices, 0, self.hidden_size:]
        # print('b_last_hidden_state--->\n', b_last_hidden_state.shape)
        last_hidden_state = torch.cat([f_last_hidden_state, b_last_hidden_state], dim=-1)
        # print('last_hidden_state--->\n', last_hidden_state.shape)

        # Dropout 和全连接层
        hidden = self.dropout(last_hidden_state)  # [batch_size, hidden_size*2]
        logits = self.fc(hidden)  # [batch_size, num_classes]

        return logits


if __name__ == '__main__':
    model = BiLSTMClassifier().to(config.device)
    print('model--->\n', model)
    model.train()

    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    for input_ids, attention_mask, labels in train_dataloader:
        y_pred = model(input_ids, attention_mask)
        print('y_pred--->\n', y_pred.shape)
        break
