
from ..src.bert_config import Config


# 继承思想
class NewConfig(Config):
    def __init__(self):
        super().__init__()
        # ========== 训练Trainer相关配置 ==========
        self.output_dir = "./training_output"  # transformers训练输出文件夹
        self.warmup_steps = 500  # 学习率预热步数（可视具体任务适当调整）
        self.weight_decay = 0.01  # 权重衰减，防止过拟合
        self.logging_steps = 100  # 多久打印一次训练日志
        self.eval_steps = 500  # 每隔多少步进行一次评估
        self.save_steps = 500  # 每隔多少步保存一次模型checkpoint
        self.save_total_limit = 2  # 只保留最近N个训练模型，防止磁盘爆满


config = NewConfig()
if __name__ == '__main__':
    print("BERT模型结构配置:\n", config.bert_config)
    # 测试分词器将中文token转换为BERT词表下的ID
    input_size = config.tokenizer.convert_tokens_to_ids(["你", "好", "中", "国", "人"])
    print("分词器ID编码示例：", input_size)
    print("类别列表：", config.class_list)
    with open(config.class_path, 'r', encoding='utf-8') as f:
        data = f.read()
    print(data)
