class Config(object):
    def __init__(self, use_char_segmentation=True):
        # 原始数据路径
        self.train_datapath = "../data/train.txt"
        self.test_datapath = "../data/test.txt"
        self.dev_datapath = "../data/dev.txt"

        # 模型路径
        self.ft_model_save_path = "./save_models"

        # 样本类别文件
        self.class_datapath = "../data/class.txt"

        # 处理完的数据（用于训练）
        self.final_data = "./final_data"

        # 是否使用字符分词
        # True: 字符分词  False: 词分词
        self.use_char_segmentation = use_char_segmentation


if __name__ == '__main__':
    conf = Config()
    print(conf.ft_model_save_path)
    print(conf.train_datapath)
