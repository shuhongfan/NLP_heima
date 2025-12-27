# -*- coding:utf-8 -*-
import jieba.posseg as pseg
from wordcloud import WordCloud
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt

def get_a_list(text):
    '''
    获得句子text中的所有形容词
    :param text: 一个句子
    :return: 列表：每个元素都是形容词
    '''
    r = []
    for value in pseg.lcut(text):
        # print(value)
        # print(value.flag)
        # print(value.word)
        if value.flag == 'a':
            r.append(value.word)
    return r

def get_word_cloud(keywords):
    '''
    画图展示词云
    :param keywords: 列表形式，存储的是高频词
    '''
    # 1.实例化词云对象
    wordcloud = WordCloud(font_path='./cn_data/SimHei.ttf', max_words=100, background_color='white')
    # 2. 获取展示词云的数据，字符串形式，空格隔开
    keywords_str = ' '.join(keywords)
    # 3. 生成词云
    wordcloud.generate(keywords_str)
    # 4.画图控制台展示
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()



# 定义主函数

def main():
    # 1.读取数据集
    train_data = pd.read_csv('./cn_data/train.tsv', sep='\t')
    print(f'train_data--》{train_data}')
    dev_data = pd.read_csv('./cn_data/dev.tsv', sep='\t')
    # 2.获取训练集中的正样本(positive)
    train_p_sentence = train_data[train_data["label"] == 1]["sentence"]
    # 3. 获得训练集正样本中的所有形容词
    p_trian_vocabs = list(chain(*map(lambda x: get_a_list(x), train_p_sentence)))
    print(f'p_trian_vocabs--》{p_trian_vocabs}')
    print(f'p_trian_vocabs--》{len(p_trian_vocabs)}')

    # 4.调用get_word_cloud方法
    get_word_cloud(p_trian_vocabs)
    print('*' * 60 )
    # 训练集负样本词云
    n_train_data = train_data[train_data['label'] == 0 ]['sentence']

    # 2 获取正样本的每个句子的形容词 p_a_train_vocab = chain(*map(a,b))
    n_a_train_vocab = list(chain(*map(lambda x: get_a_list(x) , n_train_data)))
    # print(n_a_dev_vocab)
    # print(list(n_a_dev_vocab))

    # 3 调用绘制词云函数
    get_word_cloud(n_a_train_vocab)

    # 获得验证集上正样本
    p_valid_data = dev_data[dev_data["label"] == 1]["sentence"]

    # 对正样本的每个句子的形容词
    valid_p_a_vocab = list(chain(*map(lambda x: get_a_list(x), p_valid_data)))
    # print(train_p_n_vocab)

    # 获得验证集上负样本
    n_valid_data = dev_data[dev_data["label"] == 0]["sentence"]

    # 获取负样本的每个句子的形容词
    valid_n_a_vocab = list(chain(*map(lambda x: get_a_list(x), n_valid_data)))

    # 调用绘制词云函数
    get_word_cloud(valid_p_a_vocab)
    get_word_cloud(valid_n_a_vocab)


if __name__ == '__main__':
    main()
