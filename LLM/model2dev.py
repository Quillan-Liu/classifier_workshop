import pandas as pd
from tqdm import tqdm
from rich import print
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
from LLM.model2pred import model2pred
import time

# 1.获取label2id映射字典
label2id = {label.strip(): idx for idx, label in enumerate(open('../data/class.txt', 'r', encoding='utf-8'))}
# print('label2id--->\n', label2id)


# 2.定义读取测试数据集的函数
def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', names=['text', 'label'])
    # print('data--->\n', data.head())
    return data['text'].tolist(), data['label'].astype(int).tolist()


# 3.定义评估函数
def evaluate(file_path):
    start_time = time.time()
    # 3.1 加载测试数据集
    texts, true_labels = load_data(file_path)
    # print('texts--->\n', texts)
    # print('true_labels--->\n', true_labels)

    # 3.2 准备两个空列表, 收集结果
    pred_labels = []
    results = []

    # 3.3 遍历测试数据集, 逐条预测
    for text in tqdm(texts, desc='evaluating...'):
        data_dict, _ = model2pred(text)
        # 获取预测标签名
        pred_label = data_dict['category']
        # print('pred_label--->', pred_label)
        # 将标签名转换成标签下标
        pred_idx = label2id[pred_label]
        # 保存结果
        pred_labels.append(pred_idx)
        results.append({'text': text,
                        'pred_label': pred_label,
                        'reason': data_dict['reason']})
    # 3.4 计算并打印准确率、精确率、F1值、分类评估报告
    accuracy = accuracy_score(true_labels, pred_labels)  # 计算整体准确率
    precision = precision_score(true_labels, pred_labels, average='micro')  # 使用微平均计算精确率
    f1score = f1_score(true_labels, pred_labels, average='micro')  # 使用微平均计算F1分数
    report = classification_report(true_labels, pred_labels)  # 生成详细的分类报告
    elapsed_time = (time.time() - start_time)  # 计算总耗时（秒）
    return accuracy, precision, f1score, report, elapsed_time


if __name__ == '__main__':
    accuracy, precision, f1score, report, elapsed_time = evaluate('../data/dev2.txt')
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1 Score: {f1score:.4f}')
    print(report)
    print(f'Elapsed Time: {elapsed_time:.2f} seconds')