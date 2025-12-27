import fasttext

# 训练词向量模型
def dm_fasttext_01():
    # 直接开始训练：以非监督的方式进行
    model = fasttext.train_unsupervised('./data/ai20aa')
    # 保存模型
    model.save_model('./data/ai20_fil9.bin')

# 获取某个词的词向量和检验模型效果
def dm_fasttext_02():
    # 加载模型
    model = fasttext.load_model('./data/ai20_fil9.bin')
    # 直接获取某个词的向量
    # results = model.get_word_vector("the")
    # print(type(results))
    # print(results.shape)
    # print(results)
    # 检验模型的效果
    results = model.get_nearest_neighbors("dog")
    print(f'dog的临近词-->{results}')

# 训练词向量模型:修改参数
def dm_fasttext_03():
    # 直接开始训练：以非监督的方式进行
    model = fasttext.train_unsupervised('./data/ai20aa',"cbow", dim=100, lr=0.1, epoch=1)
    # 保存模型
    model.save_model('./data/ai20_fil9_new.bin')

if __name__ == '__main__':
    # dm_fasttext_01()
    # dm_fasttext_02()
    dm_fasttext_03()