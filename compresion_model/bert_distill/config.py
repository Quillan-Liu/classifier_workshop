from bert.src.bert_config import Config


class DistillConfig(Config):
    def __init__(self):
        super().__init__()
        ####################### 以下为新增属性 #######################
        # BERT蒸馏模型存储结果路径
        # 硬标签蒸馏
        self.distill_h_model_save_path = "./save_models/student_distill_h.pt"
        # 软标签蒸馏
        self.distill_s_model_save_path = "./save_models/student_distill_s.pt"

        # BiLSTM学生模型参数配置
        self.embed_size = 256
        self.hidden_size_lstm = 512
        self.num_layers = 4
        self.dropout = 0.3


config = DistillConfig()

if __name__ == '__main__':
    print(config.bert_config)
    input_size = config.tokenizer.convert_tokens_to_ids(["你", "好", "中", "国", "人"])
    print(input_size)
