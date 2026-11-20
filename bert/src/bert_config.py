import torch
from pathlib import Path
from transformers.models import BertModel, BertTokenizer, BertConfig

class Config(object):
    def __init__(self):
        """
        配置类，包含模型和训练所需的各种参数。
        """
        self.model_name = "bert"  # 模型名称
        self.data_path = Path(__file__).resolve().parents[2] / "data"
        self.train_path = self.data_path / "train.txt"
        self.dev_path = self.data_path / "dev3.txt"
        self.test_path = self.data_path / "test.txt"

        self.class_path = self.data_path / "class.txt"
        # 标签名列表
        self.class_list = [line.strip() for line in open(self.class_path, encoding="utf-8")]  # 类别名单

        self.model_save_path = Path(__file__).resolve().parents[1] / "save_models" / "bertclassifier_model_20261116.pt"

        # 模型训练+预测的时候
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练设备，如果GPU可用，则为cuda，否则为cpu

        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 2  # epoch数
        self.batch_size = 8  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = r"D:\learning_materials\llm算法\code\_12-迁移学习\model\bert-base-chinese"  # 预训练BERT模型的路径
        self.bert_model = BertModel.from_pretrained(self.bert_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # BERT模型的分词器
        self.bert_config = BertConfig.from_pretrained(self.bert_path)  # BERT模型的配置
        self.hidden_size = 768  # BERT模型的隐藏层大小


config = Config()


if __name__ == '__main__':
    print(config.bert_config)
    print(config.bert_model)
    input_size = config.tokenizer.convert_tokens_to_ids(["你", "好", "中", "国", "人"])
    print(input_size)
    print(config.class_list)
    print(config.num_classes)
    with open(config.dev_path, 'r', encoding='utf-8') as f:
        data = f.read()
    print(data[:10])
