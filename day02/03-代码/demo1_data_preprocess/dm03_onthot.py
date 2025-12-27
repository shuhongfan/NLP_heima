# 导入keras中的词汇映射器Tokenizer
from keras.preprocessing.text import Tokenizer
# 导入用于对象保存与加载的joblib
import joblib

def get_one_hot():
    # 准确语料库
    vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
    # 实例化Tokenizer
    my_tokenizer = Tokenizer()
    my_tokenizer.fit_on_texts(vocabs)
    # 打印word_index, index_word
    print(my_tokenizer.word_index)
    # print(my_tokenizer.index_word)
    # 对每个vocab进行one-hot编码的实现
    for vocab in vocabs:
        zero_list = [0] * len(vocabs)
        idx = my_tokenizer.word_index[vocab] - 1
        zero_list[idx] = 1
        print(f'当前{vocab}的one-hot编码是{zero_list}')

    # 保存tokenizer,方便下次使用
    mypath = './mytokenizer'
    joblib.dump(my_tokenizer, mypath)
    print('模型保存成功')


def use_one_hot():
    # 加载训练好的tokenizer
    mypath = './mytokenizer'
    my_tokenizer = joblib.load(mypath)
    print(f'my_tokenizer--》{my_tokenizer.word_index}')
    token = '李宗盛'
    zero_list = [0] * len(my_tokenizer.word_index)
    idx = my_tokenizer.word_index[token] - 1
    zero_list[idx] = 1
    print(f'当前{token}的one-hot编码是{zero_list}')

if __name__ == '__main__':
    # get_one_hot()
    use_one_hot()