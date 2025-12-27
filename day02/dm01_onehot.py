import jieba
# 导入keras中的词汇映射器Tokenizer
# from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import Tokenizer
# 导入用于对象保存与加载的joblib
import joblib


def get_onehot():
    # 1.准备语料
    vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
    # 2. 实例化分词器
    tokenizer = Tokenizer()
    # 3. 训练分词器
    tokenizer.fit_on_texts(vocabs)
    # 4.检验训练效果
    print(tokenizer.word_index)
    print(tokenizer.index_word)
    # 5.查找每个单词的one-hot编码
    for vocab in vocabs:
        # 初始化一个全零的列表，长度为len(vocabs)
        zero_list = [0] * len(vocabs)
        # 找到当前单词对应的索引
        idx = tokenizer.word_index[vocab] - 1
        zero_list[idx] = 1
        print(f'当前单词{vocab}的one-hot编码是{zero_list}')

    # 6. 保存训练好的tokenizer
    my_path = './my_tokenizer'
    joblib.dump(tokenizer, my_path)
    print('tokenier已经保存')


def use_onehot():
    # 1.准备语料
    vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
    # tokenizer的保存路径
    my_path = './my_tokenizer'
    tokenizer = joblib.load(my_path)
    # 2.需要进行one-hot编码的token
    token = '王力宏'
    zero_list = [0] * len(vocabs)
    idx = tokenizer.word_index[token] - 1
    zero_list[idx] = 1
    print(f'当前：{token} 的one-hot编码是：{zero_list}')


if __name__ == '__main__':
    get_onehot()
    use_onehot()