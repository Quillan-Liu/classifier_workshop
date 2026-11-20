import fasttext


def train():
    # 1.模型训练
    model = fasttext.train_supervised('./final_data/train_fastText_word.txt',
                                      autotuneValidationFile='./final_data/dev_fastText_word.txt',  # 验证文件
                                      thread=1,  # 使用所有CPU核心
                                      verbose=3,  # 显示训练过程
                                      autotuneModelSize='100M',  # 模型体积大小
                                      )

    print(model.words[:10])  # 查看模型训练的单词列表
    print(model.get_subwords('赤壁'))  # 查看模型训练的子词
    # 2.模型评估
    result = model.test('./final_data/test_fastText_word.txt')
    print('result--->\n', result)
    # 3.模型推理
    y_pred = model.predict(text='《 赤壁 OL 》攻城战 诸侯战 硝烟 又起')
    print('y_pred--->\n', y_pred)
    # 4.模型保存
    model.save_model('save_models/fastText_word_auto.bin')


if __name__ == '__main__':
    train()
