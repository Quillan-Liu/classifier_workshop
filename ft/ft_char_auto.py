import fasttext


def train():
    # 1.模型训练
    model = fasttext.train_supervised('./final_data/train_fastText_char.txt',
                                      autotuneValidationFile='./final_data/dev_fastText_char.txt',  # 验证文件
                                      thread=1,  # 使用所有CPU核心
                                      verbose=3,  # 显示训练过程
                                      autotuneModelSize='100M',  # 模型体积大小
                                      )

    print(model.words[:10])  # 查看模型训练的单词列表
    print(model.get_subwords('赤'))  # 查看模型训练的子词
    # 2.模型评估
    result = model.test('./final_data/test_fastText_char.txt')
    print('result--->\n', result)
    # 3.模型推理
    y_pred = model.predict(text='《 赤 壁 O L 》 攻 城 战 诸 侯 战 硝 烟 又 起')
    print('y_pred--->\n', y_pred)
    # 4.模型保存
    model.save_model('save_models/fastText_char_auto.bin')


if __name__ == '__main__':
    train()