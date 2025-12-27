# -*-coding:utf-8-*-
# 用于正则表达式
import re
# 用于构建网络结构和函数的torch工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# torch中预定义的优化方法工具包
import torch.optim as optim
import time
# 用于随机生成数据
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# 指定设备windows：是否在GPU或者CPU进行训练
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 指定设备M1:
# device = torch.device('mps')
device = torch.device('cpu')
# print(device)
# 起始标志
SOS_token = 0
# 结束标志
EOS_token = 1
# 最大句子长度不能超过10个 (包含标点)
MAX_LENGTH = 10
# 数据文件路径
data_path = './data/eng-fra-v2.txt'

#todo:1 定义字符串的清洗函数
def norm_string(s):
    # s代表输入的字符串
    s1 = s.lower().strip()
    # 在.!?标点符号前加上空格
    s2 = re.sub(r'([.?!])', r' \1', s1)
    #将字符串中所有的除了大小写字母以及.!?标点符号的其他字符都用空格替换
    s3 = re.sub('[^a-zA-Z.?!]+', r' ', s2)
    return s3

#todo:2 读取文件，获取样本并且获得英文词典以及法文词典

def get_data():
    # 2.1 读取文档数据
    with open('data/eng-fra-v2.txt') as fr:
        sequences = fr.read().strip().split('\n')
    # print(f'sequences--》{len(sequences)}')
    # print(f'sequences[:2]-->{sequences[:2]}')

    # 2.2 获取pair对['i m .\tj ai ans .']-->[[i m ., j ai ans .]]-->[[英文句子,法文句子],[英文句子,法文句子],....]
    eng_fra_pairs = [[norm_string(s) for s in line.split('\t')] for line in sequences]
    # print(f'eng_fra_pairs[:2]-->{eng_fra_pairs[:2]}')

    # 2.3 遍历上述的pair对，获得英文字典以及法文字典
    # 2.3.1 获取word2index
    english_word2index = {"SOS": 0, "EOS": 1}
    english_word_n = 2
    french_word2index = {"SOS": 0, "EOS": 1}
    french_word_n = 2

    # 开始遍历
    for pair in eng_fra_pairs:
        # 构建英文词典
        # print(f'pair--》{pair}')
        for word in pair[0].split(' '):
            if word not in english_word2index:
                english_word2index[word] = english_word_n
                english_word_n += 1
        # 构建法文字典
        for word in pair[1].split(' '):
            if word not in french_word2index:
                french_word2index[word] = french_word_n
                french_word_n += 1

    # 2.3.2 获取index2word
    english_index2word = {v: k for k, v in english_word2index.items()}
    french_index2word = {v: k for k, v in french_word2index.items()}

    return english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, eng_fra_pairs

english_word2index, english_index2word, english_word_n, french_word2index, \
    french_index2word, french_word_n, eng_fra_pairs = get_data()
# todo:3 构建dataset数据源


class SeqDataset(Dataset):
    def __init__(self, eng_fre_paris):
        super().__init__()
        # 获取样本对
        self.pairs = eng_fre_paris
        # 获取样本的总量
        self.sample_len = len(eng_fre_paris)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, item):
        # 异常值修正
        item = min(max(item, 0), self.sample_len-1)
        # 根据索引取出样本
        # 取出英文句子
        x = self.pairs[item][0]
        # print(f'x------>{x}')
        # 取出法文
        y = self.pairs[item][1]
        # print(f'y------>{y}')
        # 将样本x进行张量化表示
        x2index = [english_word2index[word] for word in x.split(' ')]
        x2index.append(EOS_token) # 可加可不加（编码器阶段）
        # 将上述的结果张量化
        tensor_x = torch.tensor(x2index, dtype=torch.long, device=device)
        # print(f'tensor_x--》{tensor_x}')
        # 将y进行张量化表示
        y2index = [french_word2index[word] for word in y.split(' ')]
        y2index.append(EOS_token)# 一定加
        # 将上述的结果张量化
        tensor_y = torch.tensor(y2index, dtype=torch.long, device=device)
        return tensor_x, tensor_y

# todo:4.实例化dataloader

def get_dataloader():
    # 实例dataset对象
    seq_dataset = SeqDataset(eng_fra_pairs)
    # 实例化dataloader
    train_dataloader = DataLoader(dataset=seq_dataset,
                                  batch_size=1,
                                  shuffle=True)
    return train_dataloader

# todo:5. 定义GRU编码器
class EncoderGRU(nn.Module):
    def __init__(self, eng_vocab_size, hidden_size):
        super().__init__()
        # eng_vocab_size:英文单词的总个数，需要被embedding单词的数量
        self.eng_vocab_size = eng_vocab_size
        # hidden_size:代表单词的词嵌入维度
        self.hidden_size = hidden_size
        # 定义Embedding层
        self.embed = nn.Embedding(eng_vocab_size, hidden_size)

        # 定义GRU层:注意：这里输入和输出维度一致，并且设置了batch_first=True,意味这gru模型的输入是：【batch_size, seq_len, embedding_dim】
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, x, h0):
        # x--》来自于dataloader，形状为--》[batch_size, seq_len]-->[1, 8]
        # h0-->初始化的gru模型的隐藏层张量的结果--》[1, 1, 256]
        # 再将x送入Gru模型之前，一定要转换为三维的张量，所以x-->[1, 8]-->[1, 8, 256]
        embed_x = self.embed(x)
        # 将embed_x和h0送入gru模型
        #  output-->[1, 8, 256]; hn-->[1, 1, 256]
        output, hn = self.gru(embed_x, h0)
        return output, hn

    def inithidden(self):
        # 注意：需要将张量放到GPU上
        return torch.zeros(1, 1, self.hidden_size, device=device)


# todo: 6.定义不带attention的解码器
class DecoderGRU(nn.Module):
    def __init__(self, fre_vocab_size, hidden_size):
        super().__init__()
        # fre_vocab_size：代表：法文单词的总个数:4345
        self.fre_vocab_size = fre_vocab_size
        # hidden_size：代表：词嵌入的维度:256
        self.hidden_size = hidden_size
        # 定义Embedding 层
        self.embed = nn.Embedding(fre_vocab_size, hidden_size)
        # 定义GRU层
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # 定义输出层
        self.out = nn.Linear(hidden_size, fre_vocab_size)

    def forward(self, y0, h0):
        # y0--》来自于dataloader，形状为--》[batch_size, 1]-->[1, 1]
        # h0-->初始化的gru模型的隐藏层张量的结果--》[1, 1, 256]
        # 1.再将y0送入Gru模型之前，一定要转换为三维的张量，所以y0-->[1, 1]-->[1, 1, 256]
        embed_y0 = self.embed(y0)
        # 2.将上述的embed_y0经过relu激活函数，可以防止过拟合
        relu_y0 = F.relu(embed_y0)
        # 3.将relu之后的结果送入gru模型:output-->[1, 1, 256]
        output, hn = self.gru(relu_y0, h0)
        # 4.将gru模型的输出结果送入输出层，但是需要降维
        # result ===>[1, fre_vocab_size]-->[1, 4345]
        result = self.out(output[0])
        # 5.将上述的result进行log_softmax
        return F.log_softmax(result, dim=-1), hn

    def inithidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def test_decoder():
    # 1.获取训练数据集
    train_dataloader = get_dataloader()
    # 2.实例化encoder
    eng_vocab_size = len(english_word2index)
    hidden_size = 256
    encoder = EncoderGRU(eng_vocab_size, hidden_size)
    # 需要把模型放到GPU上
    encoder = encoder.to(device=device)
    # 3. 实例化解码器对象
    fre_vocab_size = french_word_n
    hidden_size = 256
    decoder = DecoderGRU(fre_vocab_size, hidden_size)
    decoder = decoder.to(device=device)
    # 4.开始将数据送入seq2seq架构得到结果
    for x, y in train_dataloader:
        print(f'x---》{x.shape}')
        print(f'y---》{y.shape}')
        print(f'y---》{y}')
        # 将x送入编码器得到编码器的结果
        h0 = encoder.inithidden()
        encoder_output, encoder_hidden = encoder(x, h0)
        print(f'encoder_output---》{encoder_output.shape}')
        print(f'encoder_hidden---》{encoder_hidden.shape}')
        hidden = encoder_hidden
        # 开始解码：注意，一定是一个词一个词去解码
        for idx in range(y.shape[1]):
            temp_vector = y[0][idx].view(1, -1)
            output, hidden = decoder(temp_vector, hidden)
            print(f'output--》{output.shape}')
        break


# todo: 7.定义带attention的解码器

class AttentionDecoder(nn.Module):
    def __init__(self, french_vocab_size, hidden_size, dropout_p=0.1, max_len=MAX_LENGTH):
        super().__init__()
        # frech_vocab_size：法文单词的总个数
        self.french_vocab_size = french_vocab_size
        # hidden_size：词嵌入的维度
        self.hidden_size = hidden_size
        # dropout_p：随机失活的系数
        self.dropout_p = dropout_p
        # max_len：最大句子长度
        self.max_len = max_len
        # 定义Embedding层：num_embeddings=frech_vocab_size, embedding_dim=hidden_size
        self.embed = nn.Embedding(french_vocab_size, hidden_size)
        # 定义第一个全连接层：计算注意力权重分数
        self.atten = nn.Linear(2*hidden_size, max_len)
        # 定义第二个全连接层：让注意力的结果按照指定尺寸输出
        self.atten_combin = nn.Linear(2*hidden_size, hidden_size)
        # 定义GRU层
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # 定义第三个全连接层：输出层，
        self.out = nn.Linear(hidden_size, french_vocab_size)
        # 定义随机失活层
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, Q, K, V):
        # Q代表：当前解码时，预测出的上一个单词（最开始的时候代表：SOS）:[1, 1]
        # K代表：解码器上一层的隐藏层输出结果（最开始的时候是编码器最后一个单词的隐藏层张量的结果）[1, 1, hidden_size]-->[1,1,256]
        # V代表：编码器每个时间步的隐藏层的输出结果--》但是我们规定了最大句子长度--》[max_len, hidden_size]-->[10, 256]
        # 1.需要将Q输入Embedding层: embed_x-->[1, 1, 256]
        embed_x = self.embed(Q)
        # 2. 对embed_x进行随机失活：防止过拟合dropout_x:[1, 1, 256]
        dropout_x = self.dropout(embed_x)
        # 3.按照注意力的计算步骤，实现最终注意力的计算
        # 3.1 选择第一种注意力计算规则，实现Q\K|V的运算
        # 3.1.1 将Q（dropout_x）和K进行拼接--》[1, 1, 512]-->经过Linear层(512, max_len)-->[1, 1, 10]
        atten_weight = F.softmax(self.atten(torch.cat((dropout_x, K), dim=-1)), dim=-1)
        # 3.1.2 将atten_weight[1, 1, 10]和V[10, 256]进行矩阵乘法运算得到结果:temp_vc-->[1,1,256]
        temp_vc = torch.bmm(atten_weight, V.unsqueeze(dim=0))
        # 3.2 将上述第一步计算的结果和Q(dropout_x)进行拼接:[1, 1, 512]
        cat_vc = torch.cat((dropout_x, temp_vc), dim=-1)
        # 3.3 将上述拼接之后的结果按照指定尺寸输出attention_output:[1, 1, 256]
        attention_output = F.relu(self.atten_combin(cat_vc))
        # 4. 将attention_output以及K(hidden)送入GRU模型output-->[1, 1, 256]
        output, hidden = self.gru(attention_output, K)
        # 5.将output降维送入输出层:result---[1, 4345]
        result = self.out(output[0])
        return F.log_softmax(result), hidden, atten_weight

def test_attentionDecoder():
    # 1.获取训练数据集
    train_dataloader = get_dataloader()
    # 2.实例化encoder
    eng_vocab_size = len(english_word2index)
    hidden_size = 256
    encoder = EncoderGRU(eng_vocab_size, hidden_size)
    # 需要把模型放到GPU上
    encoder = encoder.to(device=device)
    # 3. 实例化Attention解码器对象
    fre_vocab_size = french_word_n
    hidden_size = 256
    decoder = AttentionDecoder(fre_vocab_size, hidden_size)
    attention_decoder = decoder.to(device=device)
    # 4.开始将数据送入seq2seq架构得到结果
    for x, y in train_dataloader:
        print(f'x---》{x.shape}')
        print(f'y---》{y.shape}')
        print(f'y---》{y}')
        # 将x送入编码器得到编码器的结果
        h0 = encoder.inithidden()
        encoder_output, encoder_hidden = encoder(x, h0)
        # print(f'encoder_output---》{encoder_output}')
        # print(f'encoder_output---》{encoder_output.shape}')
        # print(f'encoder_hidden---》{encoder_hidden.shape}')
        # # 定义中间语意张量C
        encoder_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
        # 将真实的x编码后的结果赋值给encoder_output_c，其余多余的为0
        # encoder_output-->[1, 5, 256]
        for idx in range(encoder_output.shape[1]):
            encoder_output_c[idx] = encoder_output[0, idx]
        # print(f'encoder_output_c--》{encoder_output_c}')
        # 解码：一个token一个token去解码
        hidden = encoder_hidden
        for j in range(y.shape[1]):
            temp_vec = y[0, j].view(1,  -1)
            output, hidden, atten_weight = attention_decoder(Q=temp_vec, K=hidden, V=encoder_output_c)
            print(f'output->{output.shape}')
            print(f'hidden->{hidden.shape}')
            print(f'atten_weight->{atten_weight.shape}')



# todo:8.定义模型的训练函数
# 8.1 定义模型训练的超参数
my_lr = 1e-4
epochs = 1
teacher_forcing_ratio = 0.5



# 8.2 定义模型的训练函数

def train_seqseq():
    # 1.获取数据
    train_dataloader = get_dataloader()
    # 2.实例化编码器对象
    eng_vocab_size = english_word_n
    hidden_size = 256
    encoder = EncoderGRU(eng_vocab_size, hidden_size).to(device=device)
    # encoder = encoder.to(device=device)
    # 3.实例化带Attention的解码器对象
    french_vocab_size = french_word_n
    hidden_size = 256
    atten_decoder = AttentionDecoder(french_vocab_size, hidden_size).to(device=device)
    # 4.实例化优化器对象
    encoder_adam = optim.Adam(encoder.parameters(), lr=my_lr)
    atten_decoder_adam = optim.Adam(atten_decoder.parameters(), lr=my_lr)
    # 5.实例化损失函数对象
    cross_entropy = nn.NLLLoss()
    # 6. 定义存储损失的列表
    plot_loss_list = []
    # 7. 开始外部循环
    for epoch_idx in range(1, epochs+1):
        # 内部定义一些训练日志的参数
        print_loss_total, plot_loss_total = 0.0, 0.0
        start_time = time.time()
        # 7.1 开始内部迭代循环
        for item, (x, y) in enumerate(tqdm(train_dataloader), start=1):
            # 开始调用内部迭代的函数
            # print(f'x---.{x}')
            # print(f'y---.{y}')
            my_loss = train_iter(x, y, encoder, atten_decoder,encoder_adam, atten_decoder_adam, cross_entropy)
            # print(f'my_loss--》{my_loss}')
            print_loss_total += my_loss
            plot_loss_total += my_loss
            # 每隔1000步打印损失日志
            if item % 1000 == 0:
                # 计算平均损失
                avg_loss= print_loss_total / 1000
                # print_loss_total重新初始化为0
                print_loss_total = 0.0
                print('轮次%d  损失%.6f 时间:%d' % (epoch_idx, avg_loss, time.time() - start_time))

            # 每隔100步，保存平均损失，画图
            if item % 100 == 0:
                plot_avg_loss = plot_loss_total / 100
                plot_loss_list.append(plot_avg_loss)
                plot_loss_total = 0.0


        torch.save(encoder.state_dict(), './save_model/ai23_seq2seq_encode_%d.pth'% epochs)
        torch.save(atten_decoder.state_dict(), './save_model/ai23_seq2seq_decode_%d.pth'% epochs)
    # 8.绘图
    plt.figure(0)
    plt.plot(plot_loss_list)
    plt.savefig('ai23_seq2seq_loss.png')
    plt.show()

# 8.3 定义模型内部迭代训练函数

def train_iter(x, y, encoder, atten_decoder,encoder_adam, atten_decoder_adam, cross_entropy):
    # x---》代表英文原始数据的输入--》[batch_size, seq_len]-->[1, 6]
    # y---》代表法文原始数据的输入--》[batch_size, seq_len]-->[1, 8]
    # encoder:编码器
    # atten_decoder： 带Attention的解码器
    # encoder_adam： 编码器优化器
    # atten_decoder_adam： 带Attention的解码器优化器
    # cross_entropy：损失函数对象
    # 1. 需要将x送入编码器得到编码器之后的结果:encoder_output-->[1, 6, 256];encoder_hidden-->[1, 1, 256]
    h0 = encoder.inithidden()
    encoder_output, encoder_hidden = encoder(x, h0)
    # 2.进行解码器参数的准备:
    # 2.1 encoder_output_c-->就是代表的Value
    encoder_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    # 将真实的编码结果进行赋值
    for idx in range(x.shape[1]):
        encoder_output_c[idx] = encoder_output[0, idx]
    # print(f'encoder_output_c-->{encoder_output_c.shape}')
    # 2.2: encoder_hidden:代表解码器上一时间步的隐藏层输出，这里其实就是代表：Key
    # 这里解码器的第一个时间步的隐藏层输入用编码器的最后一个时间步的隐藏层输出结果初始化
    decoder_hidden = encoder_hidden
    # print(f'decoder_hidden--》{decoder_hidden.shape}')
    # 2.3 :定义解码器开始解码的第一个字符为SOS，代表Query
    input_y = torch.tensor([[SOS_token]], device=device)
    # print(f'input_y--》{input_y}')

    # 3.定义变量
    my_loss = 0.0
    y_len = y.shape[1]
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # 4.将数据送入解码器
    if use_teacher_forcing:
        # 真实样本法文句子有多长，就要遍历多少次
        for idx in range(y_len):
            # 获取模型预测的结果
            output_y, decoder_hidden, atten_weight = atten_decoder(Q=input_y, K=decoder_hidden, V=encoder_output_c)
            # print(f'output_y--》{output_y.shape}')
            # 获取真实的标签结果
            target_y = y[0][idx].view(1)
            my_loss = my_loss + cross_entropy(output_y, target_y)
            # 用真实的label当作下一个输入
            input_y = y[0][idx].view(1, -1)
    else:
        # 真实样本法文句子有多长，就要遍历多少次
        for idx in range(y_len):
            # 获取模型预测的结果
            output_y, decoder_hidden, atten_weight = atten_decoder(Q=input_y, K=decoder_hidden, V=encoder_output_c)
            # print(f'output_y--》{output_y.shape}')
            # 获取真实的标签结果
            target_y = y[0][idx].view(1)
            my_loss = my_loss + cross_entropy(output_y, target_y)
            topv, topi = torch.topk(output_y, k=1)
            # print(f'topi--》{topi}')
            # print(f'topi--》{topi.item()}')
            # print(f'topv--》{topv}')
            if topi.item() == EOS_token:
                break
            input_y = topi.detach()

    # 5.梯度清零
    encoder_adam.zero_grad()
    atten_decoder_adam.zero_grad()
    # 6.反向传播
    my_loss.backward()
    # 7.参数更新
    encoder_adam.step()
    atten_decoder_adam.step()
    return my_loss.item() / y_len



#todo: 9.实现评估函数的应用
# 准备好已经训练好的模型的路径
encoder_path = './stu_model/ai23_seq2seq_encode_2.pth'
decoder_path = './stu_model/ai23_seq2seq_decode_2.pth'


def seq2seq_evaluate(tensor_x, encoder_model, decoder_model):
    with torch.no_grad():
        # 1. 需要将tensor_x送入编码器得到编码器之后的结果:encoder_output-->[1, 6, 256];encoder_hidden-->[1, 1, 256]
        h0 = encoder_model.inithidden()
        encoder_output, encoder_hidden = encoder_model(tensor_x, h0)
        # 2.进行解码器参数的准备:
        # 2.1 encoder_output_c-->就是代表的Value
        encoder_output_c = torch.zeros(MAX_LENGTH, encoder_model.hidden_size, device=device)
        # 将真实的编码结果进行赋值
        for idx in range(tensor_x.shape[1]):
            encoder_output_c[idx] = encoder_output[0, idx]
        # print(f'encoder_output_c-->{encoder_output_c.shape}')
        # 2.2: encoder_hidden:代表解码器上一时间步的隐藏层输出，这里其实就是代表：Key
        # 这里解码器的第一个时间步的隐藏层输入用编码器的最后一个时间步的隐藏层输出结果初始化
        decoder_hidden = encoder_hidden
        # print(f'decoder_hidden--》{decoder_hidden.shape}')
        # 2.3 :定义解码器开始解码的第一个字符为SOS，代表Query
        input_y = torch.tensor([[SOS_token]], device=device)
        # print(f'input_y--》{input_y}')

        # 定义存储解码出法文单词的列表
        decoded_list = []
        # 定义存储解码出每个token的注意力权重的张量
        decoded_attention = torch.zeros(MAX_LENGTH, MAX_LENGTH)
        # 开始解码过程
        # idx = 0
        for idx in range(MAX_LENGTH):
            output_y, decoder_hidden, atten_weight = decoder_model(input_y, decoder_hidden, encoder_output_c)
            # print(f'output_y--》{output_y.shape}')
            # print(f'atten_weight--》{atten_weight.shape}')
            # 将每一步预测的注意力的权重进行赋值
            # decoded_attention[idx] = atten_weight[0, 0]
            decoded_attention[idx] = atten_weight
            # 获得最大预测值的概率值以及对应的索引
            topv, topi = torch.topk(output_y, k=1)
            # print(f'topi--》{topi}')
            if topi.item() == EOS_token:
                decoded_list.append('<EOS>')
                break
            else:
                decoded_list.append(french_index2word[topi.item()])
            # print(f'decoded_list--》{decoded_list}')
            input_y = topi
    return decoded_list, decoded_attention[:idx+1]



def use_evaluate():
    # 实例化编码器对象，并且加载训练好的模型参数
    eng_vocab_size = english_word_n
    hidden_size = 256
    encoder_model = EncoderGRU(eng_vocab_size, hidden_size).to(device=device)
    # encoder_model.load_state_dict(torch.load(encoder_path))
    encoder_model.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    print(f'encoder_model--》{encoder_model}')
    # for parameter in encoder_model.parameters():
    #     print(parameter.device)
    #     break

    # 实例化解码器对象，并且加载训练好的模型参数
    french_vocab_size = french_word_n
    hidden_size = 256
    decoder_model = AttentionDecoder(french_vocab_size, hidden_size).to(device=device)
    # decoder_model.load_state_dict(torch.load(decoder_path))
    decoder_model.load_state_dict(torch.load(decoder_path, map_location='cpu'))
    print(f'decoder_model--》{decoder_model}')
    # 准备测试的语料
    my_samplepairs = [['i m impressed with your french .', 'je suis impressionne par votre francais .'],
                      ['i m more than a friend .', 'je suis plus qu une amie .'],
                      ['she is beautiful like her mother .', 'elle est belle comme sa mere .']]
    print('my_samplepairs--->', len(my_samplepairs))

    # 将遍历每一个测试样本，将原始的英文句子送入模型得到预测结果，并且和真实的法文标签做对比
    for item, pair in enumerate(my_samplepairs):
        # print(f'pair--》{pair}')
        x = pair[0]
        y = pair[1]
        # 将x进行张量化
        temp_x = [english_word2index[word] for word in x.split(' ')]
        temp_x.append(EOS_token)
        tensor_x = torch.tensor(temp_x, dtype=torch.long, device=device).view(1, -1)
        # print(f'tensor_x--》{tensor_x}')
        # 将张量化的x送入评估函数
        deocder_words, attention = seq2seq_evaluate(tensor_x, encoder_model, decoder_model)
        # print(f'deocder_words--》{deocder_words}')
        # print(f'attention--》{attention.shape}')
        predit_y = ' '.join(deocder_words)
        print('最终的预测结果')
        print(f'x--->{x}')
        print(f'y--->{y}')
        print(f'predit_y--->{predit_y}')

def show_attention():
    # 实例化编码器对象，并且加载训练好的模型参数
    eng_vocab_size = english_word_n
    hidden_size = 256
    encoder_model = EncoderGRU(eng_vocab_size, hidden_size).to(device=device)
    # encoder_model.load_state_dict(torch.load(encoder_path))
    encoder_model.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    print(f'encoder_model--》{encoder_model}')
    # for parameter in encoder_model.parameters():
    #     print(parameter.device)
    #     break

    # 实例化解码器对象，并且加载训练好的模型参数
    french_vocab_size = french_word_n
    hidden_size = 256
    decoder_model = AttentionDecoder(french_vocab_size, hidden_size).to(device=device)
    # decoder_model.load_state_dict(torch.load(decoder_path))
    decoder_model.load_state_dict(torch.load(decoder_path, map_location='cpu'))
    print(f'decoder_model--》{decoder_model}')
    # 准备测试的语料
    sentence = "we are both teachers ."

    temp_x = [english_word2index[word] for word in sentence.split(' ')]
    temp_x.append(EOS_token)
    tensor_x = torch.tensor(temp_x, dtype=torch.long, device=device).view(1, -1)
    # print(f'tensor_x--》{tensor_x}')
    # 将张量化的x送入评估函数
    deocder_words, attention = seq2seq_evaluate(tensor_x, encoder_model, decoder_model)
    # print(f'deocder_words--》{deocder_words}')
    # print(f'attention--》{attention.shape}')
    predit_y = ' '.join(deocder_words)
    print('最终的预测结果')
    print(f'predit_y--->{predit_y}')
    plt.matshow(attention.detach().numpy())
    plt.savefig('ai23—_attention.png')
    plt.show()
if __name__ == '__main__':
    # eng_vocab_size = len(english_word2index)
    # hidden_size = 256
    # encoder = EncoderGRU(eng_vocab_size, hidden_size)
    # # 需要把模型放到GPU上
    # encoder = encoder.to(device=device)
    # print(encoder)
    # train_dataloader = get_dataloader()
    # # print(len(train_dataloader))
    # for x, y in train_dataloader:
    #     h0 = encoder.inithidden()
    #     output, hn = encoder(x, h0)
    #     print(f'output--》{output.shape}')
    #     print(f'hn--》{hn.shape}')
    #     break
    # 实例化解码器对象
    # fre_vocab_size = french_word_n
    # hidden_size = 256
    # decoder = DecoderGRU(fre_vocab_size, hidden_size)
    # print(decoder)
    # test_decoder()
    # french_vocab_size = french_word_n
    # hidden_size = 256
    # atten_decoder = AttentionDecoder(french_vocab_size, hidden_size)
    # print(atten_decoder)
    # test_attentionDecoder()
    # train_seqseq()
    use_evaluate()
    # show_attention()