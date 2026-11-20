import torch

from ..src.bert_utils import load_data, TextDataset
from .bert_trainer_config import config
from torch.utils.data import DataLoader


# 重新定义collate_fn函数
def collate_fn(batch):
    # print('batch--->\n', batch)
    # 1.获取x和y两组数据
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]

    # 2.使用分词器对x进行分词并转换ID 文本长度规范处理
    inputs = config.tokenizer(x,
                              return_tensors='pt',
                              padding='max_length',
                              truncation=True,
                              max_length=config.pad_size)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    # 3.将y转换成张量对象
    labels = torch.tensor(y, dtype=torch.long)
    # 返回字典类型数据, 后续在trainer中会自动解包
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


# 定义构建数据加载器函数
def build_dataloader(dataset=False):
    # 1.加载数据集
    train_data = load_data(config.train_path)
    dev_data = load_data(config.dev_path)
    test_data = load_data(config.test_path)
    # 2.构建数据集对象
    train_dataset = TextDataset(train_data)
    dev_dataset = TextDataset(dev_data)
    test_dataset = TextDataset(test_data)
    if dataset:
        return train_dataset, dev_dataset, test_dataset
    # 3.构建数据加载器对象
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=collate_fn)
    dev_dataloader = DataLoader(dataset=dev_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=collate_fn)
    return train_dataloader, dev_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    for batch in train_dataloader:
        print(batch["input_ids"].shape)
        print(batch["attention_mask"].shape)
        print(batch["labels"].shape)
        exit()
