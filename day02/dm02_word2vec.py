# coding:utf-8
import fasttext

'''
unsupervised_default = {
    'model': "skipgram",
    'lr': 0.05,
    'dim': 100,
    'ws': 5,
    'epoch': 5,
    'minCount': 5,
    'minCountLabel': 0,
    'minn': 3,
    'maxn': 6,
    'neg': 5,
    'wordNgrams': 1,
    'loss': "ns",
    'bucket': 2000000,
    'thread': multiprocessing.cpu_count() - 1,
    'lrUpdateRate': 100,
    't': 1e-4,
    'label': "__label__",
    'verbose': 2,
    'pretrainedVectors': "",
    'seed': 0,
    'autotuneValidationFile': "",
    'autotuneMetric': "f1",
    'autotunePredictions': 1,
    'autotuneDuration': 60 * 5,  # 5 minutes
    'autotuneModelSize': ""
}
'''
def dm1_use_fasttext():
    # 1.直接调用无监督训练方法训练词向量模型
    # model = fasttext.train_unsupervised('./fil9', epoch=1)
    # 2. 保存模型
    # model_path = 'ai23_fil9.bin'
    # # model.save_model(model_path)
    # # 3.加载模型，查询某个单词的向量
    # model = fasttext.load_model(model_path)
    # result = model.get_word_vector('the')
    # print(f'result--》{result.shape}')
    # print(f'result--》{type(result)}')
    # # 4.  查找"运动"的邻近单词, 我们可以发现"体育网", "运动汽车", "运动服"等.
    # result2 = model.get_nearest_neighbors('sports')
    # print(f'result2-->{result2}')
    # 修改训练模型的参数
    my_model = fasttext.train_unsupervised('./fil9', model='cbow', epoch=1, lr=0.1)





if __name__ == '__main__':
    dm1_use_fasttext()
