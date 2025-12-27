# -*- coding:utf-8 -*-
# 导入torch工具
import json

import matplotlib.pyplot as plt
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
# 进度条
from tqdm import tqdm


# todo: 1.获取常用的字符数量：也就是one-hot编码的去重之后的词汇的总量n

all_letters = string.ascii_letters + " ,;.'"
# print(f'all_letters-->{all_letters}')

n_letters = len(all_letters)
print(f'当前字符的总量--》{n_letters}')

# todo:2. 获取国家名种类数
categorys = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese',
             'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']
# 国家名个数
categorynum = len(categorys)
print('国家总数--->', categorynum)

# todo：3. 读取数据到内存
def read_data(filepath):
    # 3.1 定义两个空列表my_list_x（存储人名）, my_list_y（存储国家类型）
    my_list_x , my_list_y = [], []
    # 3.2 读取文件内容
    with open(filepath, encoding='utf-8') as fr:
        for line in fr.readlines():
            if len(line) <= 5:
                continue
            x, y = line.strip().split('\t')
            my_list_x.append(x)
            my_list_y.append(y)
    return my_list_x, my_list_y


# todo：4.构建Dataset类
class NameDataset(Dataset):
    def __init__(self, my_list_x, my_list_y):
        super().__init__()
        # 获取x
        self.my_list_x = my_list_x
        # 获取y
        self.my_list_y = my_list_y
        # 获取样本的数量
        self.sample_len = len(my_list_x)

    # 获取样本的数量
    def __len__(self):
        return self.sample_len

    # 根据索引取出元素item:代表索引
    def __getitem__(self, item):
        # 1.对异常索引进行修正
        item = min(max(item, 0), self.sample_len-1)
        # 2.根据索引取出样本
        x = self.my_list_x[item]
        # print(f'x-->{x}')
        y = self.my_list_y[item]
        # print(f'y--->{y}')
        # 3.将人名变成one-hot编码的张量形式
        # 3.1 初始化全零的张量 #(3, 57)
        tensor_x = torch.zeros(len(x), n_letters)
        # print(f'tensor_x-->{tensor_x}')
        # 3.2 遍历人名的每一个字母，进行one-hot编码的赋值
        for idx, letter in enumerate(x):
            # print(f'idx--》{idx}')
            # print(f'letter--》{letter}')
            tensor_x[idx][all_letters.find(letter)] = 1
            # print(f'tensor_x--》{tensor_x}')
        # print(f'tensor_x修改之后的：-->{tensor_x}')

        tensor_y = torch.tensor(categorys.index(y), dtype=torch.long)
        return tensor_x, tensor_y


# todo: 5.实例化dataloader对象

def get_dataloader():
    # 读取文档数据
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    # 获取dataset对象
    name_dataset = NameDataset(my_list_x, my_list_y)
    # 封装dataset得到dataloader对象: 会对数据进行增加维度
    train_dataloader = DataLoader(dataset=name_dataset,
                                  batch_size=1,
                                  shuffle=True)

    return train_dataloader

# todo: 6.定义RNN层
class NameRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        # input_size:代表输入数据的词嵌入维度
        self.input_size = input_size
        #  hidden_size:代表RNN模型输出的维度（隐藏层输出维度)
        self.hidden_size = hidden_size
        # output_size: 输出层类别的总个数：18个国家
        self.output_size = output_size
        # num_layers:RNN隐藏层的个数
        self.num_layers = num_layers

        # 定义RNN层:(默认情况下：batch_first为False)
        self.rnn = nn.RNN(self.input_size, self.hidden_size, num_layers)

        # 定义输出层
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # 定义logSoftmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, h0):
        # print(f'x--->{x.shape}')
        # print(f'h0--->{h0.shape}')
        # x:代表输入的原始数据维度[seq_len, input_size]
        # h0:代表初始化的值，[num_layers, batch_size, hidden_size]-->[1, 1, 128]
        # x需要先升维：[seq_len, input_size]--》[seq_len, batch_size, input_size]
        x1 = torch.unsqueeze(x, dim=1) # x.unsqueeze(dim=1)
        # print(f'x1-->{x1.shape}')
        # 将x1和h0送入RNN模型
        output, hn = self.rnn(x1, h0)

        # print(f'output--》{output.shape}')
        # print(f'hn--》{hn.shape}')
        # 获取最后一个单词的隐藏层张量来代表整个句子（人名）的语意
        # 这里可以直接用hn代替
        temp = output[-1] # [1, 128]
        # 将temp送入输出层:result-->[1, 18]
        result = self.out(temp)

        return self.softmax(result), hn

    def inithidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


# tood: 7.定义LSTM层
class NameLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        # input_size:代表输入数据的词嵌入维度
        self.input_size = input_size
        #  hidden_size:代表LSTM模型输出的维度（隐藏层输出维度)
        self.hidden_size = hidden_size
        # output_size: 输出层类别的总个数：18个国家
        self.output_size = output_size
        # num_layers:LSTM隐藏层的个数
        self.num_layers = num_layers

        # 定义LSTM层:(默认情况下：batch_first为False)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers)

        # 定义输出层
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # 定义logSoftmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x0, h0, c0):
        # x0:代表输入的原始数据维度[seq_len, input_size]-->[5, 57]
        # h0,c0:代表初始化的值，[num_layers, batch_size, hidden_size]-->[1, 1, 128]
        # x0需要先升维：[seq_len, input_size]--》[seq_len, batch_size, input_size]
        x1 = torch.unsqueeze(x0, dim=1)
        # 把x0,h0,c0送入lstm模型output-->[5, 1, 128]
        output, (hn, cn) = self.lstm(x1, (h0, c0))
        # 取出最后一个单词的对应的向量output[-1]-->[1, 128]
        temp = output[-1]
        # 将temp送入输出层:result-->[1, 18]
        result = self.out(temp)
        return self.softmax(result), hn, cn

    def inithidden(self):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        return h0, c0



# todo: 8.定义GRU层
class NameGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        # input_size:代表输入数据的词嵌入维度
        self.input_size = input_size
        #  hidden_size:代表RNN模型输出的维度（隐藏层输出维度)
        self.hidden_size = hidden_size
        # output_size: 输出层类别的总个数：18个国家
        self.output_size = output_size
        # num_layers:RNN隐藏层的个数
        self.num_layers = num_layers

        # 定义GRU层:(默认情况下：batch_first为False)
        self.gru = nn.GRU(self.input_size, self.hidden_size, num_layers)

        # 定义输出层
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # 定义logSoftmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, h0):
        # print(f'x--->{x.shape}')
        # print(f'h0--->{h0.shape}')
        # x:代表输入的原始数据维度[seq_len, input_size]
        # h0:代表初始化的值，[num_layers, batch_size, hidden_size]-->[1, 1, 128]
        # x需要先升维：[seq_len, input_size]--》[seq_len, batch_size, input_size]
        x1 = torch.unsqueeze(x, dim=1) # x.unsqueeze(dim=1)
        # print(f'x1-->{x1.shape}')
        # 将x1和h0送入GRU模型
        output, hn = self.gru(x1, h0)

        # print(f'output--》{output.shape}')
        # print(f'hn--》{hn.shape}')
        # 获取最后一个单词的隐藏层张量来代表整个句子（人名）的语意
        # 这里可以直接用hn代替
        temp = output[-1] # [1, 128]
        # 将temp送入输出层:result-->[1, 18]
        result = self.out(temp)

        return self.softmax(result), hn

    def inithidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

mylr = 1e-3
epochs = 1
# todo: 9 定义RNN训练的函数
def train_rnn():
    # 1.读取文档数据
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    # 2.获取dataset对象
    name_dataset = NameDataset(my_list_x, my_list_y)
    # 3.实例化模型
    input_size = 57
    hidden_size = 128
    output_size = 18
    rnn_model = NameRNN(input_size, hidden_size, output_size)
    # 4.实例化损失函数对象
    cross_entropy = nn.NLLLoss()
    # 5.实例化优化器对象
    adam = optim.Adam(rnn_model.parameters(), lr=mylr)
    # 6. 定义训练模型的打印日志的参数
    start_time = time.time() # 开始的时间
    total_iter_num = 0 # 已经训练的样本的总个数
    total_loss = 0.0 # 已经训练的样本的损失之和
    total_loss_list = [] # 每隔100个样本计算一下平均损失，画图
    total_num_acc = 0 # 已经训练的样本中预测正确的样本的个数
    total_acc_list = [] # 每隔100个样本计算一下平均准确率，画图
    # 6.开始模型的训练
    # 开始外部epoch的迭代
    for epoch_idx in range(epochs):
        # 实例化dataloader
        train_dataloader = DataLoader(dataset=name_dataset, batch_size=1, shuffle=True)
        # 开始内部数据的迭代
        for idx, (x,  y) in enumerate(tqdm(train_dataloader)):
            # print(f'x--》{x.shape}')
            # print(f'y--》{y.shape}')
            # # 将数据送入模型
            h0 = rnn_model.inithidden()
            output, hn = rnn_model(x[0], h0)
            # print(f'output-->{output}')
            # print(f'y-->{y}')
            # 计算损失:output.shape-->[1, 18];y.shape-->[1]
            my_loss = cross_entropy(output, y)

            # print(f'my_loss--》{my_loss}')
            # 梯度清零
            adam.zero_grad()
            # 反向传播: 计算梯度
            my_loss.backward()
            # 梯度更新
            adam.step()
            # 打印日志参数
            # 获取已经训练的样本的总数
            total_iter_num += 1
            # 获取已经训练的样本的总损失
            total_loss = total_loss + my_loss.item()
            # 获取已经训练的样本中预测正确的总个数
            # print(f'output--》{output}')
            # torch.argmax(output)取出概率值最大值对应的索引
            pred_idx = 1 if torch.argmax(output).item() == y.item() else 0
            # print(f'pred_idx-->{pred_idx}')
            total_num_acc = total_num_acc + pred_idx
            # 每100次训练 求一次平均损失 平均准确率
            if total_iter_num % 100 == 0:
                # 保留平均损失
                avg_loss = total_loss / total_iter_num
                total_loss_list.append(avg_loss)
                # 保留平均准确率
                avg_acc = total_num_acc / total_iter_num
                total_acc_list.append(avg_acc)
            # 每隔2000步，打印日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_num_acc / total_iter_num
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' %(epoch_idx+1, temp_loss, time.time() - start_time, temp_acc))
        # 每轮都保存一个模型
        torch.save(rnn_model.state_dict(), './save_model/ai23_rnn_%d.bin'%(epoch_idx+1))
    # 7.计算训练的总时间
    all_time = time.time() - start_time
    # 8.将训练的结果进行保存
    dict1 = {"total_loss_list": total_loss_list,
             "all_time": all_time,
             "total_acc_list": total_acc_list}
    with open('rnn_result.json', 'w') as fw:
        fw.write(json.dumps(dict1))
    # return total_loss_list, all_time, total_acc_list

# todo: 10 定义LSTM训练的函数
def train_lstm():
    # 1.读取文档数据
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    # 2.获取dataset对象
    name_dataset = NameDataset(my_list_x, my_list_y)
    # 3.实例化模型
    input_size = 57
    hidden_size = 128
    output_size = 18
    rnn_model = NameLSTM(input_size, hidden_size, output_size)
    # 4.实例化损失函数对象
    cross_entropy = nn.NLLLoss()
    # 5.实例化优化器对象
    adam = optim.Adam(rnn_model.parameters(), lr=mylr)
    # 6. 定义训练模型的打印日志的参数
    start_time = time.time() # 开始的时间
    total_iter_num = 0 # 已经训练的样本的总个数
    total_loss = 0.0 # 已经训练的样本的损失之和
    total_loss_list = [] # 每隔100个样本计算一下平均损失，画图
    total_num_acc = 0 # 已经训练的样本中预测正确的样本的个数
    total_acc_list = [] # 每隔100个样本计算一下平均准确率，画图
    # 6.开始模型的训练
    # 开始外部epoch的迭代

    for epoch_idx in range(epochs):
        # 实例化dataloader

        train_dataloader = DataLoader(dataset=name_dataset, batch_size=1, shuffle=True)
        # 开始内部数据的迭代
        for idx, (x,  y) in enumerate(tqdm(train_dataloader)):
            # print(f'x--》{x.shape}')
            # print(f'y--》{y.shape}')
            # # 将数据送入模型
            h0, c0 = rnn_model.inithidden()
            output, hn, cn = rnn_model(x[0], h0, c0)

            # 计算损失:output.shape-->[1, 18];y.shape-->[1]
            my_loss = cross_entropy(output, y)

            # print(f'my_loss--》{my_loss}')
            # 梯度清零
            adam.zero_grad()
            # 反向传播: 计算梯度
            my_loss.backward()
            # 梯度更新
            adam.step()
            # 打印日志参数
            # 获取已经训练的样本的总数
            total_iter_num += 1
            # 获取已经训练的样本的总损失
            total_loss = total_loss + my_loss.item()
            # 获取已经训练的样本中预测正确的总个数
            # print(f'output--》{output}')
            # torch.argmax(output)取出概率值最大值对应的索引
            pred_idx = 1 if torch.argmax(output).item() == y.item() else 0
            # print(f'pred_idx-->{pred_idx}')
            total_num_acc = total_num_acc + pred_idx
            # 每100次训练 求一次平均损失 平均准确率
            if total_iter_num % 100 == 0:
                # 保留平均损失
                avg_loss = total_loss / total_iter_num
                total_loss_list.append(avg_loss)
                # 保留平均准确率
                avg_acc = total_num_acc / total_iter_num
                total_acc_list.append(avg_acc)
            # 每隔2000步，打印日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_num_acc / total_iter_num
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' %(epoch_idx+1, temp_loss, time.time() - start_time, temp_acc))
        # 每轮都保存一个模型
        torch.save(rnn_model.state_dict(), './save_model/ai23_lstm_%d.bin'%(epoch_idx+1))
    # 7.计算训练的总时间
    all_time = time.time() - start_time
    # 8.将训练的结果进行保存
    dict1 = {"total_loss_list": total_loss_list,
             "all_time": all_time,
             "total_acc_list": total_acc_list}
    with open('lstm_result.json', 'w') as fw:
        fw.write(json.dumps(dict1))
    # return total_loss_list, all_time, total_acc_list

# todo: 11 定义GRU训练的函数
def train_gru():
    # 1.读取文档数据
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    # 2.获取dataset对象
    name_dataset = NameDataset(my_list_x, my_list_y)
    # 3.实例化模型
    input_size = 57
    hidden_size = 128
    output_size = 18
    rnn_model = NameGRU(input_size, hidden_size, output_size)
    # 4.实例化损失函数对象
    cross_entropy = nn.NLLLoss()
    # 5.实例化优化器对象
    adam = optim.Adam(rnn_model.parameters(), lr=mylr)
    # 6. 定义训练模型的打印日志的参数
    start_time = time.time() # 开始的时间
    total_iter_num = 0 # 已经训练的样本的总个数
    total_loss = 0.0 # 已经训练的样本的损失之和
    total_loss_list = [] # 每隔100个样本计算一下平均损失，画图
    total_num_acc = 0 # 已经训练的样本中预测正确的样本的个数
    total_acc_list = [] # 每隔100个样本计算一下平均准确率，画图
    # 6.开始模型的训练
    # 开始外部epoch的迭代
    for epoch_idx in range(epochs):
        # 实例化dataloader
        train_dataloader = DataLoader(dataset=name_dataset, batch_size=1, shuffle=True)
        # 开始内部数据的迭代
        for idx, (x,  y) in enumerate(tqdm(train_dataloader)):
            # print(f'x--》{x.shape}')
            # print(f'y--》{y.shape}')
            # # 将数据送入模型
            h0 = rnn_model.inithidden()
            output, hn = rnn_model(x[0], h0)
            # print(f'output-->{output}')
            # print(f'y-->{y}')
            # 计算损失:output.shape-->[1, 18];y.shape-->[1]
            my_loss = cross_entropy(output, y)

            # print(f'my_loss--》{my_loss}')
            # 梯度清零
            adam.zero_grad()
            # 反向传播: 计算梯度
            my_loss.backward()
            # 梯度更新
            adam.step()
            # 打印日志参数
            # 获取已经训练的样本的总数
            total_iter_num += 1
            # 获取已经训练的样本的总损失
            total_loss = total_loss + my_loss.item()
            # 获取已经训练的样本中预测正确的总个数
            # print(f'output--》{output}')
            # torch.argmax(output)取出概率值最大值对应的索引
            pred_idx = 1 if torch.argmax(output).item() == y.item() else 0
            # print(f'pred_idx-->{pred_idx}')
            total_num_acc = total_num_acc + pred_idx
            # 每100次训练 求一次平均损失 平均准确率
            if total_iter_num % 100 == 0:
                # 保留平均损失
                avg_loss = total_loss / total_iter_num
                total_loss_list.append(avg_loss)
                # 保留平均准确率
                avg_acc = total_num_acc / total_iter_num
                total_acc_list.append(avg_acc)
            # 每隔2000步，打印日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_num_acc / total_iter_num
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' %(epoch_idx+1, temp_loss, time.time() - start_time, temp_acc))
        # 每轮都保存一个模型
        torch.save(rnn_model.state_dict(), './save_model/ai23_gru_%d.bin'%(epoch_idx+1))
    # 7.计算训练的总时间
    all_time = time.time() - start_time
    # 8.将训练的结果进行保存
    dict1 = {"total_loss_list": total_loss_list,
             "all_time": all_time,
             "total_acc_list": total_acc_list}
    with open('gru_result.json', 'w') as fw:
        fw.write(json.dumps(dict1))
    # return total_loss_list, all_time, total_acc_list


# todo:12 绘图对比不同模型的性能

def compare_rnns():
    # 1.读取rnn模型的训练结果
    with open('rnn_result.json', 'r')as fr:
        rnn_dict = json.loads(fr.read())
    # print(f'rnn_dict--->{rnn_dict}')
    # print(f'rnn_dict--->{type(rnn_dict)}')
    # 2.读取lstm模型的训练结果
    with open('lstm_result.json', 'r') as fr:
        lstm_dict = json.loads(fr.read())

    # 3.读取gru模型的训练结果
    with open('gru_result.json', 'r') as fr:
        gru_dict = json.loads(fr.read())

    # 4. 绘图
    # 4.1 绘制损失对比曲线图
    plt.figure(0)
    plt.plot(rnn_dict["total_loss_list"], label='RNN')
    plt.plot(lstm_dict["total_loss_list"], label='LSTM', color='red')
    plt.plot(gru_dict["total_loss_list"], label='GRU', color='blue')
    plt.legend(loc='upper left')
    plt.savefig('./ai23_avg_loss.png')
    plt.show()

    # 4.2 绘制柱状图对比时间
    plt.figure(1)
    x_data = ["RNN", "LSTM", "GRU"]
    y_data = [rnn_dict["all_time"], lstm_dict["all_time"], gru_dict["all_time"]]
    plt.bar(range(len(x_data)), y_data, tick_label=x_data)
    plt.savefig("./ai23_time.png")
    plt.show()

    # 4.3 绘制准确率对比图
    plt.figure(2)
    plt.plot(rnn_dict["total_acc_list"], label='RNN')
    plt.plot(lstm_dict["total_acc_list"], label='LSTM', color='red')
    plt.plot(gru_dict["total_acc_list"], label='GRU', color='blue')
    plt.legend(loc='upper left')
    plt.savefig('./ai23_avg_acc.png')
    plt.show()


def test_dataset():
    # 读取文档数据
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    # 获取dataset对象
    name_dataset = NameDataset(my_list_x, my_list_y)
    print(len(name_dataset))
    # print(name_dataset.__len__())
    tensor_x, tensor_y = name_dataset[0]
    print(f'tensor_x--》{tensor_x}')
    print(f'tensor_y--》{tensor_y}')



# todo:13.定义将人名转换为向量的函数

def name2tensor(x):
    '''
    将x转换成one-hot编码的向量形式
    :param x: "bai"
    :return:[[0,1,...], [..], [..]]
    '''
    # 1 初始化全零的张量 #(3, 57)
    tensor_x = torch.zeros(len(x), n_letters)
    # print(f'tensor_x-->{tensor_x}')
    # 2 遍历人名的每一个字母，进行one-hot编码的赋值
    for idx, letter in enumerate(x):
        # print(f'idx--》{idx}')
        # print(f'letter--》{letter}')
        tensor_x[idx][all_letters.find(letter)] = 1
        # print(f'tensor_x--》{tensor_x}')
    return tensor_x


# todo: 14.定义rnn模型的预测函数
def rnn_predict(x):
    # 1.将x--》人名转换为向量
    tensor_x = name2tensor(x)
    # 2. 实例化模型并加载训练好的模型参数
    input_size = 57
    hidden_size = 128
    output_size = 18
    rnn_model = NameRNN(input_size, hidden_size, output_size)
    rnn_model.load_state_dict(torch.load('./save_model/ai23_rnn_1.bin'))

    # 3. 预测
    with torch.no_grad():
        # 将数据送入模型
        h0 = rnn_model.inithidden()
        output, hn = rnn_model(tensor_x, h0)
        print(f'output--》{output}')
        # 取出预测结果中topk3
        topv, topi = torch.topk(output, k=3, dim=1)
        print(f'topv--》{topv}')
        print(f'topi--》{topi}')
        print(f'rnn预测的结果')
        for i in range(3):
            tempv = topv[0][i]
            tempi = topi[0][i]
            str_class = categorys[tempi]
            print(f'当前的人名是：{x}, 预测值是：{tempv:.2f}, 预测真实国家是：{str_class}')

# todo: 15.定义lstm模型的预测函数
def lstm_predict(x):
    # 1.将x--》人名转换为向量
    tensor_x = name2tensor(x)
    # 2. 实例化模型并加载训练好的模型参数
    input_size = 57
    hidden_size = 128
    output_size = 18
    rnn_model = NameLSTM(input_size, hidden_size, output_size)
    rnn_model.load_state_dict(torch.load('./save_model/ai23_lstm_1.bin'))

    # 3. 预测
    with torch.no_grad():
        # 将数据送入模型
        h0, c0 = rnn_model.inithidden()
        output, hn, cn = rnn_model(tensor_x, h0, c0)
        print(f'output--》{output}')
        # 取出预测结果中topk3
        topv, topi = torch.topk(output, k=3, dim=1)
        print(f'topv--》{topv}')
        print(f'topi--》{topi}')
        print(f'rnn预测的结果')
        for i in range(3):
            tempv = topv[0][i]
            tempi = topi[0][i]
            str_class = categorys[tempi]
            print(f'当前的人名是：{x}, 预测值是：{tempv:.2f}, 预测真实国家是：{str_class}')


# todo: 16.定义gru模型的预测函数
def gru_predict(x):
    # 1.将x--》人名转换为向量
    tensor_x = name2tensor(x)
    # 2. 实例化模型并加载训练好的模型参数
    input_size = 57
    hidden_size = 128
    output_size = 18
    rnn_model = NameGRU(input_size, hidden_size, output_size)
    rnn_model.load_state_dict(torch.load('./save_model/ai23_gru_1.bin'))

    # 3. 预测
    with torch.no_grad():
        # 将数据送入模型
        h0 = rnn_model.inithidden()
        output, hn = rnn_model(tensor_x, h0)
        print(f'output--》{output}')
        # 取出预测结果中topk3
        topv, topi = torch.topk(output, k=3, dim=1)
        print(f'topv--》{topv}')
        print(f'topi--》{topi}')
        print(f'rnn预测的结果')
        for i in range(3):
            tempv = topv[0][i]
            tempi = topi[0][i]
            str_class = categorys[tempi]
            print(f'当前的人名是：{x}, 预测值是：{tempv:.2f}, 预测真实国家是：{str_class}')

if __name__ == '__main__':
    # train_dataloader = get_dataloader()
    # model = NameRNN(input_size, hidden_size, output_size)
    # # model = NameLSTM(input_size, hidden_size, output_size)
    # model = NameGRU(input_size, hidden_size, output_size)
    # print(model)
    # for x, y in train_dataloader:
    #     # x.shape--》[batch_size, seq_len, input_size],因为batch_size=1
    #     h0 = model.inithidden()
    #     output, hn = model(x[0], h0)
    #     print(f'output--》{output.shape}')
    #     print(f'hn--》{hn.shape}')
    #     break
    # train_rnn()
    # train_lstm()
    # train_gru()
    # compare_rnns()
    # result = name2tensor(x="bai")
    # print(result)
    rnn_predict(x="zhang")
    print('*'*80)
    lstm_predict(x="zhang")
    print('*'*80)
    gru_predict(x="zhang")
















