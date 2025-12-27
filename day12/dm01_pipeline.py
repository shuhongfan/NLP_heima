# -*-coding:utf-8-*-
import torch
from transformers import pipeline

#todo:1.完成文本分类任务
def dm_test_classification():
    # 情感分类任务：sentiment-analysis或者text-classification
    # 调用piepline方法，返回的是模型的对象
    model = pipeline(task='text-classification', model='/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/chinese_sentiment')
    # 直接使用模型去预测
    # result = model("我爱北京天安门，天安门上太阳升")
    result = model("我爱你")
    print(f'文本分类result-->{result}')


#todo:2.完成特征抽取任务

def dm_test_extrcat_feature():
    # 调用piepline方法，返回的是模型的对象
    model = pipeline(task='feature-extraction', model='/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/bert-base-chinese')
    # 直接使用模型去预测
    result = model("我爱北京天安门，天安门上太阳升")
    # result = model("我爱你")
    # 预训练模型输出的结果，会默认在句子前后加上两个特殊的token，一个是CLS,一个SEP
    print(f'特征抽取result-->{type(result)}')
    print(f'特征抽取result-->{torch.tensor(result).shape}')


#todo:3.完成完形填空任务

def dm_test_fill_mask():
    # 调用piepline方法，返回的是模型的对象
    model = pipeline(task='fill-mask', model='/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/chinese-bert-wwm')
    # 直接使用模型去预测
    result = model("我想明天去[MASK]家吃饭。")
    # 预训练模型输出的结果，会默认在句子前后加上两个特殊的token，一个是CLS,一个SEP
    print(f'完形填空result-->{result}')
    # input = "我想明天去[MASK]家吃[MASK]。"
    # for i in range(2):
    #     output = model(input)
    #     print(f'完形填空result-->{output}')
    #     if type(output[0]) == list:
    #         input = ''.join(output[0][0]["sequence"].split(' ')[1:-1])
    #     else:
    #         break
    #     print(f'input--》{input}')
    #     print('*'*80)


#todo:4.完成阅读理解（QA）任务

def dm_test_qa():
    # 调用piepline方法，返回的是模型的对象
    model = pipeline(task='question-answering', model='/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/chinese_pretrain_mrc_roberta_wwm_ext_large')
    # 准备语料
    context = '我叫张三，我是一个程序员，我的喜好是打篮球。'
    questions = ['我是谁？', '我是做什么的？', '我的爱好是什么？']
    # 直接使用模型去预测
    result = model(context=context, question=questions)
    print(f'阅读理解result-->{result}')


#todo:5.完成文本摘要任务

def dm_test_summary():
    # 调用piepline方法，返回的是模型的对象
    model = pipeline(task='summarization', model='/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/distilbart-cnn-12-6')
    # 准备语料
    # 3 准备文本 送给模型
    text = "BERT is a transformers model pretrained on a large corpus of English data " \
           "in a self-supervised fashion. This means it was pretrained on the raw texts " \
           "only, with no humans labelling them in any way (which is why it can use lots " \
           "of publicly available data) with an automatic process to generate inputs and " \
           "labels from those texts. More precisely, it was pretrained with two objectives:Masked " \
           "language modeling (MLM): taking a sentence, the model randomly masks 15% of the " \
           "words in the input then run the entire masked sentence through the model and has " \
           "to predict the masked words. This is different from traditional recurrent neural " \
           "networks (RNNs) that usually see the words one after the other, or from autoregressive " \
           "models like GPT which internally mask the future tokens. It allows the model to learn " \
           "a bidirectional representation of the sentence.Next sentence prediction (NSP): the models" \
           " concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to " \
           "sentences that were next to each other in the original text, sometimes not. The model then " \
           "has to predict if the two sentences were following each other or not."

    # 直接使用模型去预测
    result = model(text)
    print(f'文本摘要result-->{result}')

#todo:6.完成NER任务

def dm_test_ner():
    # token-classification或者ner
    # 调用piepline方法，返回的是模型的对象
    model = pipeline(task='token-classification', model='/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/roberta-base-finetuned-cluener2020-chinese')
    # 准备语料
    text = "我是张三, 我来自北京"
    # 直接使用模型去预测
    result = model(text)
    print(f'ner的result-->{result}')

if __name__ == '__main__':
    # dm_test_classification()
    # dm_test_extrcat_feature()
    # dm_test_fill_mask()
    # dm_test_qa()
    # dm_test_summary()
    dm_test_ner()