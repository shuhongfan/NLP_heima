# -*-coding:utf-8-*-
# 导入工具包
import torch
import random
import torch.nn as nn
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from transformers import AdamW
# import torch.optim as optim
# optim.AdamW
from torch.utils.data import DataLoader, Dataset
import time
from tqdm import tqdm

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)
# 加载分词器
bert_tokenizer = BertTokenizer.from_pretrained("/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/bert-base-chinese")
# print(bert_tokenizer.get_vocab())
# print(bert_tokenizer.mask_token)
# print(bert_tokenizer.mask_token_id)
print(bert_tokenizer.vocab_size)
# 加载model
bert_model = BertModel.from_pretrained("/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/bert-base-chinese")
# print(my_model)
# 如果用gpu，需要把预训练模型也要放到gpu上
bert_model = bert_model.to(device)



# 自定义Dataset对象
class NspDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        # 加载数据
        dataset = load_dataset('csv', data_files=data_path, split='train')
        # 只获取样本长度大于44的样本
        self.dataset = dataset.filter(lambda x: len(x["text"]) > 44)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # 因为是要做NSP任务：给你两句话，判断第二句话是否是第一句话的真实的下一句，因此我们要构造样本对
        #（Seq1, Seq2）--》标签可以为0那就是没有关系，标签可以为1那就是有关系
        label = 1
        sequence = self.dataset[item]["text"]
        # print(f'sequence--》{sequence}')
        seq1 = sequence[:22]
        seq2 = sequence[22:44]
        # 还要有一半的概率是负样本
        if random.randint(0, 1) == 0:
            # 重新选择一个索引
            j = random.randint(0, len(self.dataset)-1)
            seq2 = self.dataset[j]["text"][22:44]
            label = 0

        return seq1, seq2, label


def collate_fn(data):
    # data--》[(seq1, seq2, label),...]
    # print(data)
    # 取出每个样本的句子对
    sequences = [value[:2] for value in data]
    # 取出每个样本的标签
    labels = [value[-1] for value in data]
    # print(f'sequences--》{sequences}')
    # print(f'labels--》{labels}')
    # 对上述原始的句子对进行编码
    inputs = bert_tokenizer.batch_encode_plus(sequences, padding='max_length', truncation=True,
                                              max_length=50, return_tensors='pt')

    # print(f'inputs--》{inputs}')

    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    attention_mask = inputs["attention_mask"]
    labels_y = torch.tensor(labels, dtype=torch.long)

    return input_ids, token_type_ids, attention_mask, labels_y
def test_dataloader():
    # 实例化dataset对象
    train_dataset = NspDataset('./data/train.csv')
    #  实例化dataloader对象
    train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn, shuffle=True, drop_last=True)
    return train_dataloader

# todo:3.定义模型

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 因为是微调，所以我们承接的结果是bert预训练模型的输出结果，为768特征,所以输出层输入特征为768，输出特征为2，（二分类）
        self.out = nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # 将上述三个输入参数送入bert预训练模型，但是注意，预训练模型在这里参数不进行更新，因此需要：with torch.no_grad():
        with torch.no_grad():
            bert_output = bert_model(input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask)
        # last_hidden_state-->[4, 200, 768]
        # pooler_output-=->[4, 768]
        # print(f'bert_output1--》{bert_output["last_hidden_state"].shape}')
        # print(f'bert_output2--》{bert_output["pooler_output"].shape}')
        # bert_output["pooler_output"]代表每个样本的CLS--token对应的隐藏层输出结果，代表整个句子的语意
        # 将bert编码之后的结果pooler_output送入输出层
        result = self.out(bert_output["pooler_output"])
        return result

# todo:4.训练模型

def model2train():
    # 第一步读文件获取数据：
    train_dataset = NspDataset(data_path='./data/train.csv')
    # print(f'train_dataset---》{train_dataset}')
    # 第二步:将上述的dataset进行再次封装
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8,
                                  collate_fn=collate_fn, shuffle=True,
                                  drop_last=True)
    # 第三步：实例化模型
    my_model = MyModel().to(device)
    # 第四步：实例化优化器对象
    my_adamw = AdamW(my_model.parameters(), lr=5e-4)
    # 第五步：实例化损失函数对象
    my_cross = nn.CrossEntropyLoss()
    # 第六步：强调预训练模型的参数不参与更新
    for param in bert_model.parameters():
        param.requires_grad_(False)

    # 定义训练的轮次
    my_model.train()
    epochs = 1
    # 开始训练
    for epoch in range(epochs):
        # 开始时间
        start_time = int(time.time())
        # 开始内部数据的迭代
        for idx, (input_ids, token_type_ids, attention_mask, labels_y) in enumerate(tqdm(train_dataloader),start=1):
            # 将数据送入模型得到预测的结果
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_y = labels_y.to(device)
            output = my_model(input_ids, token_type_ids, attention_mask)
            # 计算损失
            my_loss = my_cross(output, labels_y)
            # print(f'my_loss-》{my_loss}')
            # 梯度清零
            my_adamw.zero_grad()
            # 反向传播
            my_loss.backward()
            # 梯度更新
            my_adamw.step()

            # 每隔5步打印训练日志
            if idx % 20 == 0:
                # 取出一个批次样本中模型预测的结果
                predicts = torch.argmax(output, dim=-1)
                # 计算平均准确率
                avg_acc = (predicts == labels_y).sum().item() / len(labels_y)
                print('轮次:%d 迭代数:%d 损失:%.6f 准确率%.3f 时间%d' \
                      %(epoch, idx, my_loss.item(), avg_acc, (int)(time.time())-start_time))

        # 每轮都保存模型
        torch.save(my_model.state_dict(), './save_model/ai23_nsp_%d.bin' % (epoch+1))

# todo:5.模型测试
def model2test():
    # 第一步读文件获取数据测试集：
    test_dataset = NspDataset(data_path='./data/test.csv')
    # print(f'test_dataset---》{test_dataset}')
    # 第二步:将上述的dataset进行再次封装
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=8,
                                  collate_fn=collate_fn, shuffle=True,
                                  drop_last=True)
    # 第三步：加载训练好的模型
    my_model = MyModel().to(device)
    my_model.load_state_dict(torch.load('./save_model/ai23_nsp_1.bin'))

    # 第四步：定义测试的超参数
    total = 0 # 计算已经迭代样本
    acc_num = 0 # 计算已经预测正确的样本的个数
    # 注意，把模型设置为eval()
    my_model.eval()
    # 第五步：开始测试
    for idx, (input_ids, token_type_ids, attention_mask, labels_y) in enumerate(tqdm(test_dataloader), start=1):
        # print(f'input_ids--》{input_ids}')
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels_y = labels_y.to(device)
        with torch.no_grad():
            output = my_model(input_ids, token_type_ids, attention_mask)

        # 计算预测正确的样本个数
        predicts = torch.argmax(output, dim=-1)
        acc_num = acc_num + (predicts == labels_y).sum().item()
        total = total + len(labels_y)
        # 每个5步，打印一下平均准确率，并且取出一个批次的第一个样本，进行结果的展示
        if idx % 5 == 0:
            print(f'平均准确率：{acc_num/total}', end='  ')
            print(f'取出样本：{bert_tokenizer.decode(input_ids[0], skip_special_tokens=True)}', end='  ')
            print(f'预测值：{predicts[0]}, 真实值：{labels_y[0]}')
            print('*'*80)

if __name__ == '__main__':
    # model2train()
    model2test()