# -*-coding:utf-8-*-
import torch
from transformers import BertTokenizer, BertForMaskedLM
# todo: 1.完形填空的任务

def dm_test_fill_mask():
    # 1.加载分词器
    my_tokenizer = BertTokenizer.from_pretrained('/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/chinese-bert-wwm')
    # 2.加载模型
    my_model = BertForMaskedLM.from_pretrained('/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/chinese-bert-wwm')
    print(f'my_model--》{my_model}')
    # 3.准备语料
    sentence = "我想明天去[MASK]家吃饭."
    # 4.需要将原始的语句进行tokenzier:直接返回张量的结果
    tensor_x = my_tokenizer.encode_plus(sentence, return_tensors='pt')
    print(f'tensor_x-->{tensor_x}')
    # 预测时候，如果用到类预训练模型：加上eval()
    my_model.eval()
    # # # # 5.将上述的tensor_x[1, 12]送入模型
    # # output = my_model(input_ids=tensor_x["input_ids"],
    # #                   token_type_ids=tensor_x["token_type_ids"],
    # #                   attention_mask=tensor_x["attention_mask"])
    output = my_model(**tensor_x)
    print(f'完形填空的结果--》{output}')
    print(output["logits"].shape)
    # output["logits"].shape-->[1, 12, 21128]————》需要从这个结果中找到[MASK]对应位置输出的[21128]的概率值
    # 取出该位置的预测概率值
    tem_vector = output["logits"][:, 6]
    print(f'tem_vector--》{tem_vector.shape}')
    idx = torch.argmax(tem_vector, dim=-1)
    print(f'idx====》{idx}')
    print(my_tokenizer.convert_ids_to_tokens(idx))

if __name__ == '__main__':
    dm_test_fill_mask()