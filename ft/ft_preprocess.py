import os
import jieba
from ft.ft_config import Config

conf = Config(use_char_segmentation=True)
# 获取文件路径
class_file = conf.class_datapath
data_files = [conf.train_datapath, conf.test_datapath, conf.dev_datapath]
# 是否使用字符分词
use_char_segmentation = conf.use_char_segmentation

# 步骤 1：读取类别文件，生成 ID 到类别名的映射
id2name = {}
with open(class_file, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        id2name[idx] = line.strip()

# 循环遍历处理源文件数据
for data_file in data_files:
    # 步骤 2：检查文件是否存在
    if not os.path.exists(class_file) or not os.path.exists(data_file):
        print("类别文件及原始数据路径：", class_file, data_file)
        print("文件不存在，请检查文件是否存在！")
        break
    # 步骤 3：fastText训练数据构造
    # 功能：读取源数据文件，初始化空列表 datas，准备处理每行数据
    # 输入：源数据文件，
    # 示例：
    # 中华女子学院：本科层次仅1专业招男生    3
    # 两天价网站背后重重迷雾：做个网站究竟要多少钱    4
    # 输出：datas，处理好的样本列表
    datas = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 去除换行符和空白
            if not line:
                continue  # 跳过空行

            # 步骤 3.1：分割文本和标签，转换标签
            # 功能：分割每行数据的文本和标签，将标签数值转为 FastText 格式 (__label__<class_name>)
            # 输入：一行数据，如 "中华女子学院：本科层次仅1专业招男生 3"
            # 输出：text = "中华女子学院：本科层次仅1专业招男生", label_name = "__label__education"
            text, label = line.split('\t')  # 以制表符分割
            label_name = f"__label__{id2name[int(label)]}"  # 转换标签，如 __label__education

            # 步骤 3.2：分词处理
            # 功能：对文本进行分词（默认字符分词），用空格拼接
            # 输入：text = "中华女子学院：本科层次仅1专业招男生"
            # 输出：text_processed = "中 华 女 子 学 院 本 科 层 次 仅 1 专 业 招 男 生"
            # use_char_segmentation: True->字符分词 False->jieba 词级分词
            text = text.replace('：', '')  # 移除冒号
            words = list(text) if use_char_segmentation else jieba.lcut(text)  # 字符分词或词级分词
            text_processed = ' '.join([word.strip() for word in words])  # 拼接分词结果

            # 步骤 3.3：构造 FastText 格式
            # 功能：组合标签和分词文本，生成 FastText 格式，存入 datas
            # 输入：label_name = "__label__education", text_processed = "中 华 女 子 学 院 ..."
            # 输出：fasttext_line = "__label__education 中 华 女 子 学 院 ..."
            fasttext_line = f"{label_name} {text_processed}"
            datas.append(fasttext_line)  # 添加到列表

    # 步骤 4：设置预处理后文件保存路径
    if use_char_segmentation:
        # 文件写入路径
        if "train" in data_file:
            output_file = conf.final_data + '/train_fastText_char.txt'
        elif "test" in data_file:
            output_file = conf.final_data + '/test_fastText_char.txt'
        elif "dev2" in data_file:
            output_file = conf.final_data + '/dev2_fastText_char.txt'
        else:
            output_file = conf.final_data + '/dev_fastText_char.txt'
    else:
        if "train" in data_file:
            output_file = conf.final_data + '/train_fastText_jieba.txt'
        elif "test" in data_file:
            output_file = conf.final_data + '/test_fastText_jieba.txt'
        else:
            output_file = conf.final_data + '/dev_fastText_jieba.txt'

    # 步骤 5：保存到文件
    # 功能：将处理后的数据写入文件中，每行以换行符结束
    # 输入：datas = ['__label__education 中 华 女 子 学 院 ...', ...]
    # 输出：txt格式文件
    # 示例：
    #  __label__education 中 华 女 子 学 院 本 科 层 次 仅 1 专 业 招 男 生
    #  __label__entertainment 两 天 价 网 站 背 后 重 重 迷 雾 做 个 网 站 究 竟 要 多 少 钱
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in datas:
            f.write(line + '\n')  # 写入每行
    print("前 5 行:", datas[:5])
    print(f"数据已保存到 {output_file}")