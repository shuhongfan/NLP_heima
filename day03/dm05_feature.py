# -*- coding:utf-8 -*-

# 获取n_gram特征
n_gram = 2

def dm01_nGram(input_list):
    alist = [input_list[i:] for i in range(n_gram)]
    print(alist)
    return set(zip(*alist))

# 句子长度规范
from keras.preprocessing import sequence

sequence_len = 10

def padding(inputs):
    # padding="pre"：补齐的时候默认在前面补齐，如果想在后面补齐：padding="post"
    # truncating="pre"：截断的时候默认在前面补齐，如果想在后面截断：truncating="post"
    return sequence.pad_sequences(inputs, sequence_len, padding='post', truncating='post')


def my_padding(inputs):
    alist = []
    for value in inputs:
        if len(value) >= sequence_len:
            alist.append(value[:sequence_len])
        else:
            value1 = value + [0]*(sequence_len - len(value))
            alist.append(value1)
    return alist

if __name__ == '__main__':
    # input_list = [1, 3, 2, 1, 5, 3]
    # result = dm01_nGram(input_list)
    # print(result)
    x_train = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
               [2, 32, 1, 23, 1]]
    result = padding(x_train)
    print(result)
    result1 = my_padding(x_train)
    print(result1)
