"""
加载数据集
构建数据集对象
构建数据加载器对象
"""
import torch
from typing import List, Tuple
try:
    from .bert_config import config
except ImportError:
    from bert_config import config
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# todo:1-加载数据集, 保存到列表中
def load_data(file_path: str) -> List[Tuple[str, int]]:
    """
    加载数据集
    :param file_path: 文件路径
    :return: 内存数据 [(text, label), (text, label), ...]
    """
    data = []
    # 打开数据文件
    with open(file_path, 'r', encoding='utf-8') as f:
        # 按行处理数据
        for line in tqdm(f, desc='Loading data...'):
            # print('line--->\n', line)
            # 去掉行数据空白符
            line = line.strip()
            # print('line--->\n', line)
            # 如果行数据为空, 跳过
            if not line:
                continue
            # 将行数据用\t进行分隔
            text, label = line.split('\t')
            # print('text--->\n', text)
            # print('label--->\n', type(label), label)
            # 将text->x 和 label->y 保存到元组中, 元组保存到列表中
            data.append((text, int(label)))
    return data


# todo:2-构建数据集对象
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        根据索引获取数据
        :param item: 样本数据的索引
        :return: 样本数据的x 和 y
        """
        # 修正索引范围
        item = min(max(0, item), len(self.data) - 1)
        # print('item--->\n', item)
        # x = self.data[item][0]
        # y = self.data[item][1]
        # return x, y
        return self.data[item]


# todo:3-构建数据加载器对象
def collate_fn(batch):
    """
    批处理函数
    :param batch: 批次样本, 必写参数, 没有就报错  [(text, label), (text, label), ...]
    :return: input_ids->词下标, attention_mask->注意力掩码, labels->真实标签
    """
    # print('batch--->\n', batch)
    # 获取批次样本的x 和  y
    x = [item[0] for item in batch]
    y = []
    for item in batch:
        # print('item--->\n', type(item), item)
        y.append(item[1])
    # print('x--->\n', x)
    # print('y--->\n', y)

    # 使用分词器对象对x进行编码, 文本长度规范处理
    # data = config.tokenizer(x, padding=True)  # 只补齐, 不进行截断, 根据批次样本的最大长度进行补齐
    # print('data--->\n', data)
    data = config.tokenizer(x,
                            padding='max_length',
                            truncation=True,
                            max_length=config.pad_size,
                            return_tensors='pt')
    data.to(config.device)
    # print('data--->\n', data)
    input_ids = data.input_ids
    attention_mask = data['attention_mask']
    # print('input_ids--->\n', input_ids)
    # print('attention_mask--->\n', attention_mask)
    # 将y转换成张量对象
    labels = torch.tensor(data=y, device=config.device)
    return input_ids, attention_mask, labels


def build_dataloader():
    # 加载数据集
    train_data = load_data(config.train_path)
    dev_data = load_data(config.dev_path)
    test_data = load_data(config.test_path)
    # 构建数据集对象
    train_dataset = TextDataset(train_data)
    dev_dataset = TextDataset(dev_data)
    test_dataset = TextDataset(test_data)
    # 构建数据加载器对象
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=collate_fn,  # 接收函数名, 将批次样本交给函数进行处理
                                  )

    dev_dataloader = DataLoader(dataset=dev_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                drop_last=True,
                                collate_fn=collate_fn,  # 接收函数名, 将批次样本交给函数进行处理
                                )

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 drop_last=True,
                                 collate_fn=collate_fn,  # 接收函数名, 将批次样本交给函数进行处理
                                 )
    return train_dataloader, dev_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    print('train_dataloader--->\n', train_dataloader, len(train_dataloader))
    for input_ids, attention_mask, labels in train_dataloader:
        print('input_ids--->', input_ids.shape)
        print('attention_mask--->', attention_mask.shape)
        print('labels--->', labels.shape)
        exit()
