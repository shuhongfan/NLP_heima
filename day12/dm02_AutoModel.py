# -*-coding:utf-8-*-
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForMaskedLM, AutoModelForQuestionAnswering
from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification
# todo: 1.文本分类的任务

def dm_test_classification():
    # 1.加载分词器
    my_tokenizer = AutoTokenizer.from_pretrained('/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/chinese_sentiment')
    # 2.加载模型
    my_model = AutoModelForSequenceClassification.from_pretrained('/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/chinese_sentiment')
    print(f'my_model--》{my_model}')
    # 3.准备语料
    sentence = '人生该如何起头'
    # 4.需要将原始的语句进行tokenzier
    # cls-->101, sep-->102
    # result = my_tokenizer.encode(sentence, padding="max_length", truncation=True, max_length=20)
    # print(result)
    # 4.1.将上述my_tokenizer.encode(sentence)结果变成张量
    # tensor_x = torch.tensor([result])

    # 4.2直接返回张量的结果
    # padding="max_length":按照最大句子长度补齐；
    # truncation=True：按照最大句子长度截断
    # return_tensors='pt'：返回的是张量
    # max_length：设定的最大句子长度
    tensor_x = my_tokenizer.encode(sentence, return_tensors='pt', padding="max_length", truncation=True, max_length=20)
    print(f'tensor_x-->{tensor_x}')
    # 预测时候，如果用到类预训练模型：加上eval()
    my_model.eval()
    # # 5.将上述的tensor_x[1, 20]送入模型
    output = my_model(tensor_x)
    # output = my_model(tensor_x, return_dict=False)
    print(f'文本分类的结果--》{output}')
    print(output["logits"])
    print(output.logits)
    # 预测最终类别对应的索引
    idx = torch.argmax(output.logits, dim=-1).item()
    print(idx)


# todo: 2.特征提取的任务

def dm_test_extract_feature():
    # 1.加载分词器
    my_tokenizer = AutoTokenizer.from_pretrained('/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/bert-base-chinese')
    # 2.加载模型
    my_model = AutoModel.from_pretrained('/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/bert-base-chinese')
    print(f'my_model--》{my_model}')
    # 3.准备语料
    sentence = ['你是谁', '人生该如何起头']
    # 4.需要将原始的语句进行tokenzier:直接返回张量的结果
    # padding="max_length":按照最大句子长度补齐；
    # truncation=True：按照最大句子长度截断
    # return_tensors='pt'：返回的是张量
    # max_length：设定的最大句子长度
    # tensor_x = my_tokenizer.encode(sentence, return_tensors='pt', padding="max_length", truncation=True, max_length=20)
    # print(f'tensor_x-->{tensor_x}')
    # print(f'tensor_x-->{tensor_x.shape}')

    tensor_x = my_tokenizer.encode_plus(sentence, return_tensors='pt', padding="max_length", truncation=True, max_length=20)
    print(f'tensor_x-->{tensor_x}')
    # 预测时候，如果用到类预训练模型：加上eval()
    my_model.eval()
    # # # 5.将上述的tensor_x[1, 20]送入模型
    # output = my_model(input_ids=tensor_x["input_ids"],
    #                   token_type_ids=tensor_x["token_type_ids"],
    #                   attention_mask=tensor_x["attention_mask"])
    output = my_model(**tensor_x)
    print(f'文本特征提取的结果--》{output}')
    print(output["last_hidden_state"].shape)
    print(output["pooler_output"].shape)


# todo: 3.完形填空的任务

def dm_test_fill_mask():
    # 1.加载分词器
    my_tokenizer = AutoTokenizer.from_pretrained('/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/chinese-bert-wwm')
    # 2.加载模型
    my_model = AutoModelForMaskedLM.from_pretrained('/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/chinese-bert-wwm')
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



# todo: 4.阅读理解的任务

def dm_test_qa():
    # 1.加载分词器
    my_tokenizer = AutoTokenizer.from_pretrained('/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/chinese_pretrain_mrc_roberta_wwm_ext_large')
    # 2.加载模型
    my_model = AutoModelForQuestionAnswering.from_pretrained('/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/chinese_pretrain_mrc_roberta_wwm_ext_large')
    print(f'my_model--》{my_model}')
    # 3.准备语料
    context = '我叫张三 我是一个程序员 我的喜好是打篮球'
    questions = ['我是谁？', '我是做什么的？', '我的爱好是什么？']
    # 4.需要一个问题一个问题去解答
    my_model.eval()
    for question in questions:
        tensor_x = my_tokenizer.encode_plus(question, context, return_tensors='pt')
        # # # # 5.将上述的tensor_x[1, 12]送入模型
        # # output = my_model(input_ids=tensor_x["input_ids"],
        # #                   token_type_ids=tensor_x["token_type_ids"],
        # # #                   attention_mask=tensor_x["attention_mask"])
        # print(f'tensor_x["input_ids"]--》{tensor_x["input_ids"].shape}')
        # print(f'tensor_x["input_ids"]--》{tensor_x["input_ids"]}')
        output = my_model(**tensor_x)
        # print(f'output--》{output}')
        # print(f'output["start_logits"]-->{output["start_logits"].shape}')
        # 找到问题对应答案的开始索引位置以及结束索引位置
        start_idx = torch.argmax(output["start_logits"], dim=-1)
        end_idx = torch.argmax(output["end_logits"], dim=-1)
        # 根据上述的start_idx以及end_idx进行切片
        answer_ids = tensor_x["input_ids"][0][start_idx: end_idx+1]
        # print(f'answer_ids--》{answer_ids}')
        answer = ''.join(my_tokenizer.convert_ids_to_tokens(answer_ids))
        print(f'原始的问题是：{question},对应的答案是:--》{answer}')
        print("*"*80)


# todo: 5.文本摘要的任务

def dm_test_summary():
    # 1.加载分词器
    my_tokenizer = AutoTokenizer.from_pretrained('/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/distilbart-cnn-12-6')
    # 2.加载模型
    my_model = AutoModelForSeq2SeqLM.from_pretrained('/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/distilbart-cnn-12-6')
    # print(f'my_model--》{my_model}')
    # 3.准备语料
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
    # 4.对上述的文本进行tokenzier
    inputs = my_tokenizer.encode_plus(text, return_tensors="pt")
    print(f'inputs--》{inputs}')
    print(f'inputs--》{inputs["input_ids"].shape}')
    # 5.将数据送入模型
    my_model.eval()
    output = my_model.generate(inputs["input_ids"])
    # output = my_model.generate(**inputs)
    print(f'output--》{output}')
    print(f'output--》{output.shape}')
    # skip_special_tokens=True，去除特殊的字符；clean_up_tokenization_spaces=False：标点符合和单词分隔开
    result = my_tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(f'result--》{result}')



# todo: 6.NER的任务

def dm_test_ner():
    # 1.加载分词器
    my_tokenizer = AutoTokenizer.from_pretrained('/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/roberta-base-finetuned-cluener2020-chinese')
    # 2.加载模型
    my_model = AutoModelForTokenClassification.from_pretrained('/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/roberta-base-finetuned-cluener2020-chinese')
    # print(my_model)
    # 3.加载该模型的配置文件
    my_config = AutoConfig.from_pretrained('/Users/ligang/PycharmProjects/NLP/NLPBase/05transferlearning/model/roberta-base-finetuned-cluener2020-chinese')
    print(f'my_config--》{my_config}')
    # print(f'my_model--》{my_model}')
    # # 3.准备语料
    # text=''
    # # 4.对上述的文本进行tokenzier
    # inputs = my_tokenizer.encode_plus(text, return_tensors="pt")
    # print(f'inputs--》{inputs}')
    # print(f'inputs--》{inputs["input_ids"].shape}')
    # # 5.将数据送入模型
    # my_model.eval()
    # output = my_model.generate(inputs["input_ids"])
    # # output = my_model.generate(**inputs)
    # print(f'output--》{output}')
    # print(f'output--》{output.shape}')
    # # skip_special_tokens=True，去除特殊的字符；clean_up_tokenization_spaces=False：标点符合和单词分隔开
    # result = my_tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # print(f'result--》{result}')
    # 4.基于my_tokenizer处理数据
    inputs = my_tokenizer.encode_plus("鲁迅先生的代表作《朝花夕拾》", return_tensors='pt')
    print(f'inputs--》{inputs}')
    # 5.将inputs送入模型
    my_model.eval()
    output = my_model(**inputs)
    # print(f'output-->{output}')
    print(f'output-->{output["logits"].shape}')
    # 6.将input_ids对齐原来的token
    original_tokens = my_tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    print(f'original_tokens--》{original_tokens}')
    print(f'all_special_tokens:{my_tokenizer.all_special_tokens}')
    # 定义output_list=[]，存储每个单词及对应标签的结果
    output_list = []
    # 7. 每个token(除了特殊字符外)，都要实现标签的预测
    for token, value in zip(original_tokens, output["logits"][0]):
        if token in my_tokenizer.all_special_tokens:
            continue
        idx = torch.argmax(value, dim=-1).item()
        output_list.append((token, my_config.id2label[idx]))
    print(f'output_list--》{output_list}')




if __name__ == '__main__':
    # dm_test_classification()
    # dm_test_extract_feature()
    # dm_test_fill_mask()
    # dm_test_qa()
    # dm_test_summary()
    dm_test_ner()