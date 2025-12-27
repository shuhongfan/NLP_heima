# 导入torch工具
import torch
# 导入nn准备构建模型
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 导入torch的数据源 数据迭代器工具包
from  torch.utils.data import Dataset, DataLoader
# 用于获得常见字母及字符规范化
import string
# 导入时间工具包
import time
# 引入制图工具包
import matplotlib.pyplot as plt
# 从io中导入文件打开方法
from io import open


all_letters = string.ascii_letters+" .,;'"
n_letters = len(all_letters)
# print('n_letters:', n_letters)

# 国家名 种类数
categorys = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese',
             'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']
# 国家名 个数
categorynum = len(categorys)
# print('categorys--->', categorys)


def read_data(filePath):
    my_list_x,my_list_y = [],[]

    with open(filePath, 'r',encoding='utf-8') as f:
        readlines = f.readlines()
        for line in readlines:
            if len(line)<=5:
                continue
            (x,y)=line.strip().split('\t')
            my_list_x.append(x)
            my_list_y.append(y)
    return my_list_x,my_list_y


class NameDataSet(Dataset):
    def __init__(self, my_list_x, my_list_y):
        self.my_list_x = my_list_x
        self.my_list_y = my_list_y
        self.sample_len = len(self.my_list_x)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, idx):
        idx = min(max(idx,0),self.sample_len-1)
        x  =self.my_list_x[idx]
        y=self.my_list_y[idx]

        tensor_x = torch.zeros(len(x),n_letters)
        # print(f'tensor_x->{tensor_x}')

        for id,letter in enumerate(x):
            # print(f'id-->{id}')
            # print(f'letter->{letter}')
            tensor_x[id][all_letters.find(letter)] = 1
            # print(f'tensor_x->{tensor_x}')
        tensor_y = torch.tensor(categorys.index(y),dtype=torch.long)
        return tensor_x,tensor_y

def get_dataloader():
    my_list_x,my_list_y = read_data('./data/name_classfication.txt')
    name_dataset=NameDataSet(my_list_x,my_list_y)
    train_loader = DataLoader(name_dataset,batch_size=1,shuffle=True)

    return train_loader

class NameRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers=1):
        super().__init__()
        # 输入数据的词嵌入维度
        self.input_size = input_size
        # 隐藏输出层维度
        self.hidden_size = hidden_size
        # 、输出层类别总个数
        self.output_size=output_size

        # 定义RNN层
        self.rnn=nn.RNN(self.input_size,self.hidden_size,num_layers)

        self.out = nn.Linear(self.hidden_size,self.output_size)

        self.softmax = nn.LogSoftmax(dim=-1)


if __name__ == '__main__':
    my_list_x,my_list_y =  read_data(filePath='./data/name_classfication.txt')
    name_dataset = NameDataSet(my_list_x,my_list_y)
    # print(len(name_dataset))
    # print(name_dataset.__len__())
    tensor_x, tensor_y= name_dataset[0]
    # print(f'tensor_x->{tensor_x}')
    # print(f'tensor_y->{tensor_y}')
    name_rnn = NameRNN(57,128,18)
    print(f'RNN:{name_rnn}')
