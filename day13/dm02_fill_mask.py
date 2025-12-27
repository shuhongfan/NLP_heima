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
# print(bert_tokenizer.get_vocab())
# print(bert_tokenizer.mask_token)
# print(bert_tokenizer.mask_token_id)
print(bert_tokenizer.vocab_size)
# 加载model
bert_model = BertModel.from_pretrained("/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/bert-base-chinese")
# print(my_model)
# 如果用gpu，需要把预训练模型也要放到gpu上
bert_model = bert_model.to(device)


def collate_fn(data):
    # data是从dataset里面获取了一个批次的样本，格式为列表，列表中的每个元素是个字典，包含：{"label","text"}
    # print(data)
    sequences = [value["text"] for value in data]

    # 进行tokenizer编码
    inputs = bert_tokenizer.batch_encode_plus(sequences, padding="max_length",
                                              truncation=True, max_length=32, return_tensors='pt')

    # print(f'inputs--》{inputs}')
    # input_ids-->shape-->[8, 32]
    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    attention_mask = inputs["attention_mask"]
    # print(f'input_ids[:,16]-->{input_ids[:, 16].shape}')
    # 取出第16个单词.clone()必须，相当于深copy
    labels = input_ids[:, 16].clone()
    # 把原始输入的每个样本的第16个索引位置的值替换为[MASK]--》这里对应的也是它的索引
    # input_ids[:, 16] = bert_tokenizer.get_vocab()[bert_tokenizer.mask_token]
    input_ids[:, 16] = bert_tokenizer.mask_token_id
    # print(f'input_ids--》{input_ids}')
    # print(f'labels--》{labels}')
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, token_type_ids, attention_mask, labels

# 定义测试函数：分析dataloader

def dm_test_dataloader():
    # 1.加载数据集
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    # print(f'train_dataset--》{train_dataset}')
    # print(f'train_dataset--》{train_dataset[1]}')

    # 2. 对数据进行处理，只获取text大于32的文本
    new_train_dataset = train_dataset.filter(lambda x: len(x["text"]) > 32)
    # print(f'new_train_dataset--》{new_train_dataset}')
    # print(f'new_train_dataset--》{len(new_train_dataset[1]["text"])}')
    # 3.将上述的数据进行dataloader的封装
    train_dataloader = DataLoader(dataset=new_train_dataset, batch_size=8,
                                  shuffle=True, collate_fn=collate_fn, drop_last=True)
    # # 4.遍历dataloader
    # for input_ids, token_type_ids, attention_mask, labels in train_dataloader:
    #     print('这是测试')
    #     print(f'input_ids--》{input_ids.shape}')
    #     print(f'token_type_ids--》{token_type_ids.shape}')
    #     print(f'attention_mask--》{attention_mask.shape}')
    #     print(f'labels--》{labels.shape}')
    #     break
    return train_dataloader

# 定义模型
class  MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # bert_tokenizer.vocab_size = 21128
        self.out = nn.Linear(768, bert_tokenizer.vocab_size)

    def forward(self, input_ids, token_type_ids, attention_mask):
        with torch.no_grad():
            bert_output = bert_model(input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask)

        # print(f'bert_output--》{bert_output["last_hidden_state"].shape}')
        # bert_output["last_hidden_state"].shape->[8, 32, 768]
        # 只取出第16个位置对应的张量送入输出层得到预测的结果:output-->[8, 21128]
        output = self.out(bert_output["last_hidden_state"][:, 16])
        return output

# todo:4.训练模型

def model2train():
    # 第一步读文件获取数据：
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    # print(f'train_dataset---》{train_dataset}')
    # 需要获取样本长度大于32的样本
    new_train_dataset = train_dataset.filter(lambda x: len(x["text"])>32)
    # 第二步:将上述的dataset进行再次封装
    train_dataloader = DataLoader(dataset=new_train_dataset, batch_size=8,
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

    # 定义模型为训练模式
    my_model.train()
    # 定义训练的轮次
    epochs = 3
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
        torch.save(my_model.state_dict(), './save_model/ai23_fill_mask_%d.bin' % (epoch+1))


# todo:5.模型测试
def model2test():
    # 第一步读文件获取数据测试集：
    test_dataset = load_dataset('csv', data_files='./data/test.csv', split='train')
    # print(f'test_dataset---》{test_dataset}')
    new_test_dataset  = test_dataset.filter(lambda x: len(x["text"])>32)
    # 第二步:将上述的dataset进行再次封装
    test_dataloader = DataLoader(dataset=new_test_dataset, batch_size=8,
                                  collate_fn=collate_fn, shuffle=True,
                                  drop_last=True)
    # 第三步：加载训练好的模型
    my_model = MyModel().to(device)
    my_model.load_state_dict(torch.load('./save_model/ai23_fill_mask_3.bin'))

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
        print(f'')
        if idx % 5 == 0:
            print(f'平均准确率：{acc_num/total}', )
            print(f'取出一个样本：{bert_tokenizer.decode(input_ids[0])}',)
            print(f'预测值：{bert_tokenizer.decode(predicts[0])}, 真实值：{bert_tokenizer.decode(labels_y[0])}')
            print('*'*80)


if __name__ == '__main__':
    # model2train()
    model2test()