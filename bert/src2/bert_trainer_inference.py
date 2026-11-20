import torch
from pathlib import Path
from .bert_trainer_classifier_model import BertClassifierModel
from .bert_trainer_config import config

# ===============================
#          文本分类推理脚本
# 本模块用于加载训练好的BERT分类模型，针对单条文本进行类别预测（推理）；
# 支持通过predict(data)函数进行接口调用。
# ===============================

# 1. 加载配置对象(Config类负责所有全局超参数、路径、类别名、分词器等)
device = config.device  # 设备(cuda/cpu)
tokenizer = config.tokenizer  # BERT分词器

# 2. 实例化模型并加载训练得到的权重
model_path = Path(__file__).resolve().parent / "training_output" / "checkpoint-45000"
model = BertClassifierModel.from_pretrained(model_path)
model.to(device)
model.eval()  # 设置为评估模式


def predict(data):
    """
    针对单条数据进行BERT文本分类预测，逻辑与训练/验证阶段推理完全一致。
    本方法适合在线服务、接口/前端传单条文本时的类别预测场景。

    参数说明:
        data (dict):  必填。输入数据字典，格式要求为 {"text": "待分类的文本内容"}。
                      其中 "text" 必须为非空字符串，否则直接返回None结果。

    返回:
        dict:  返回的预测结果字典，示例:
                {
                    "text": <原始文本>,
                    "pred_class": <预测类别名称, str 或 None>
                }
              注意若输入无效，将直接输出pred_class=None。

    预测过程说明:
        1. 对输入文本做合法性校验（类型、非空）。
        2. 使用与训练阶段完全一致的分词器编码步骤，将文本编码为BERT输入格式（自动加special tokens，pad到conf.pad_size）。
        3. 利用加载好的模型，关闭梯度进行推理，取logits最大对应类别。
        4. 返回与输入文本和预测的类别标号对应的中文名称。
    """
    # 1. 提取和校验文本
    text = data.get("text", "")  # 从data字典拿出"text"
    if not isinstance(text, str) or not text.strip():  # 非字符串或仅空白，直接返回空预测
        return {"text": text, "pred_class": None}

    # 2. 分词和编码为BERT输入格式（与训练时完全相同）
    #    encode_plus返回input_ids/attention_mask，自动pad以达到模型最大长度
    encoded = tokenizer(
        text,
        max_length=config.pad_size,  # 保证最大长度一致
        truncation=True,  # 若超长则截断
        padding="max_length",  # 不足补pad
        return_tensors="pt",  # 返回PyTorch tensor格式
    )
    input_ids = encoded["input_ids"].to(device)  # shape: [1, seq_len]
    attention_mask = encoded["attention_mask"].to(device)  # shape: [1, seq_len]

    # 3. 使用torch.no_grad禁用梯度，提升推理效率
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # print('outputs--->\n', outputs)
        logits = outputs.logits
        pred_idx = torch.argmax(logits, dim=1).item()  # 获取最大值下标
        pred_class = config.class_list[pred_idx]  # 转为类别名称

    # 4. 组装输出
    return {"text": text, "pred_class": pred_class}


if __name__ == "__main__":
    # 测试示例
    sample_data = {"text": "中华女子学院：本科层次仅1专业招男生"}  # 示例输入数据
    result = predict(sample_data)
    print("预测结果：\n", result)
