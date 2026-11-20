"""
模型量化
动态量化 -> 模型推理时 计算 每层权重值或激活值 的 S和Z, 然后进行量化计算
当今时代, 算力越来越强大, 可以不用对bert系列模型进行量化, 直接使用即可
如果公司算力有限, 或者是 2022年之前, 可以对bert系列模型进行量化
"""
from bert.src.bert_classifier_model import BertClassifierModel
import torch
from bert.src.bert_utils import build_dataloader
from bert.src.bert_config import config
from bert.src.bert_train import model2dev

# 在类的外部修改对象的属性值
# 对象名.属性名 = 属性值
config.device = 'cpu'

# 1.加载训练好的模型
model = BertClassifierModel().to(config.device)
model.load_state_dict(torch.load(r'D:\tmf\bert\save_models\bertclassifier_model_20261116.pt'))
model.eval()
# print('model--->\n', model)

# 2.对模型进行动态量化
quantized_model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
print('quantized_model--->\n', quantized_model)
# 统计量化后的模型参数大小
print(
    f"8-bit 量化后的模型内存: {sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024 ** 2:.2f} MB")

# 3.评估量化后的模型
# 获取测试集数据加载器
train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
report, f1score, accuracy, precision = model2dev(quantized_model, test_dataloader)
print(f"report:\n{report}")
print(f"f1score:\n{f1score}")
print(f"accuracy:\n{accuracy}")
print(f"precision:\n{precision}")

report, f1score, accuracy, precision = model2dev(model, test_dataloader)
print(f"report:\n{report}")
print(f"f1score:\n{f1score}")
print(f"accuracy:\n{accuracy}")
print(f"precision:\n{precision}")
# 4.保存量化后的模型
torch.save(quantized_model.state_dict(), './save_models/bertclassifier_model_20261117_quantized.pt')
