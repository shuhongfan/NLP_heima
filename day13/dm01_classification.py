# -*-coding:utf-8-*-
# 导入工具包
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from transformers import AdamW
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'mps'
print(device)
# 加载分词器
bert_tokenizer = BertTokenizer.from_pretrained("/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/bert-base-chinese")
# 加载model
bert_model = BertModel.from_pretrained("/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/bert-base-chinese")
# print(my_model)
# 如果用gpu，需要把预训练模型也要放到gpu上
bert_model = bert_model.to(device)
print(bert_model)

#todo:1.读取数据

def read_data():
    # 1.读取训练数据集
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split="train")
    # print(f'train_dataset--》{train_dataset}')
    # print(f'train_dataset的样本长度--》{len(train_dataset)}')
    # print(f'train_dataset根据索引取出一个样本--》{train_dataset[0]}')
    # print(f'train_dataset根据索引切片取出多个样本--》{train_dataset[0:3]}')
    # 2.读取测试数据集
    test_dataset = load_dataset('csv', data_files='./data/test.csv', split='train')
    # print(f'test_dataset--》{test_dataset}')
    # print(f'test_dataset的样本长度--》{len(test_dataset)}')
    # print(f'test_dataset根据索引取出一个样本--》{test_dataset[0]}')
    # print(f'test_dataset根据索引切片取出多个样本--》{test_dataset[0:3]}')

    # 3.读取验证数据集
    valid_dataset = load_dataset('csv', data_files='./data/validation.csv', split='train')
    # print(f'valid_dataset--》{valid_dataset}')
    # print(f'valid_dataset的样本长度--》{len(valid_dataset)}')
    # print(f'valid_dataset根据索引取出一个样本--》{valid_dataset[0]}')
    # print(f'valid_dataset根据索引切片取出多个样本--》{valid_dataset[0:3]}')

    return train_dataset, test_dataset, valid_dataset


def collate_fn(data):
    '''
    自定义函数，目的是对dataset中的数据进行处理
    :param data:
    :return:
    '''
    # print(f'自定义函数的参数data数据展示--》{len(data)}')
    # 获取一个批次样本中的所有的句子
    sentences = [value["text"] for value in data]
    # print(f'sentences数据展示--》{sentences}')
    # print(f'sentences[0]数据展示--》{len(sentences[0])}')
    # print(f'sentences[1]数据展示--》{len(sentences[1])}')
    # print(f'sentences[2]数据展示--》{len(sentences[2])}')
    # print(f'sentences[3]数据展示--》{len(sentences[3])}')
    # 获取一个批次样本中的所有的标签
    labels = [value["label"] for value in data]
    # print(f'labels数据展示--》{labels}')
    # 对一个批次的原始句子进行张量的转换，一定要对齐长度
    inputs = bert_tokenizer.batch_encode_plus(sentences,
                                            padding='max_length',
                                            truncation=True,
                                            max_length=200,
                                            return_tensors='pt',
                                            )
    # print(f'inputs-->{inputs}')
    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    attention_mask = inputs["attention_mask"]
    labels_y = torch.tensor(labels, dtype=torch.long)

    return input_ids, token_type_ids, attention_mask, labels_y

# todo:2.获取dataloader

def get_dataloader():
    # 获取dataset
    train_dataset, _, _ = read_data()

    # 对上述的train_dataset进行dataloader的封装
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=4,
                                  collate_fn=collate_fn,
                                  drop_last=True,
                                  shuffle=True)

    # 一定要迭代train_dataloader，才能查验collate_fn
    # for input_ids, token_type_ids, attention_mask, labels_y in train_dataloader:
    #     print(f'input_ids---》{input_ids.shape}')
    #     print(f'token_type_ids---》{token_type_ids.shape}')
    #     print(f'attention_mask---》{attention_mask.shape}')
    #     print(f'labels_y---》{labels_y.shape}')
    #     print('这是测试')
    #     break
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
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
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
    epochs = 1
    my_model.train()
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
            if idx % 5 == 0:
                # 取出一个批次样本中模型预测的结果
                predicts = torch.argmax(output, dim=-1)
                # 计算平均准确率
                avg_acc = (predicts == labels_y).sum().item() / len(labels_y)
                print('轮次:%d 迭代数:%d 损失:%.6f 准确率%.3f 时间%d' \
                      %(epoch, idx, my_loss.item(), avg_acc, (int)(time.time())-start_time))

        # 每轮都保存模型
        torch.save(my_model.state_dict(), './save_model/ai23_classify_%d.bin' % (epoch+1))

# todo:5.模型测试
def model2test():
    # 第一步读文件获取数据测试集：
    test_dataset = load_dataset('csv', data_files='./data/test.csv', split='train')
    # print(f'test_dataset---》{test_dataset}')
    # 第二步:将上述的dataset进行再次封装
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=8,
                                  collate_fn=collate_fn, shuffle=True,
                                  drop_last=True)
    # 第三步：加载训练好的模型
    my_model = MyModel().to(device)
    my_model.load_state_dict(torch.load('./save_model/ai23_classify_1.bin'))

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
    # model2test()
    ...