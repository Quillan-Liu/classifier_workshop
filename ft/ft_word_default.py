import fasttext


def train():
    # 1.模型训练
    model = fasttext.train_supervised('./final_data/train_fastText_word.txt',
                                      minn=1,
                                      maxn=3)
    print(model.words[:10])  # 查看模型训练的单词列表
    print(model.get_subwords('赤壁 OL'))  # 查看模型训练的子词
    # 2.模型评估
    result = model.test('./final_data/test_fastText_word.txt')
    print('result--->\n', result)
    # 3.模型推理
    y_pred = model.predict(text='《 赤壁 OL 》攻城战 诸侯战 硝烟 又起')
    print('y_pred--->\n', y_pred)
    # 4.模型保存
    model.save_model('save_models/fastText_word_default.bin')


if __name__ == '__main__':
    train()
