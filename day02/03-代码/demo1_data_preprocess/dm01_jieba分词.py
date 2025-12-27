# coding:utf-8
import jieba

#todo: 01-jieba精确模式分词

def demo1_jieba():
    # 待切分的文本
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
    # # jieba的精确模式分词  cut 方式# 默认 cut_all=False
    # result1 = jieba.cut(content, cut_all=False) # generator object
    # print(f'result1--》{result1}')
    # 从 generator object里面取元素用next()方法
    # print(next(result1))
    # for循环方法
    # for vaue in result1:
    #     print(vaue)
    # 强制转换对象
    # print(list(result1))
    # jieba的精确模式分词 luct方式# 默认 cut_all=False
    result2 = jieba.lcut(content, cut_all=False)
    print(f'精确模式分词result2--》{result2}')


#todo: 02-jieba全模式分词
def demo2_jieba():
    # 待切分的文本
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
    # # jieba的精确模式分词  cut 方式# 默认 cut_all=False
    # result1 = jieba.cut(content, cut_all=True) # generator object
    # print(f'全模式分词result1--》{result1}')
    # # 从 generator object里面取元素用next()方法
    # # print(next(result1))
    # # for循环方法
    # for vaue in result1:
    #     print(vaue)
    # 强制转换对象
    # print(list(result1))
    # jieba的精确模式分词 luct方式# 默认 cut_all=False
    result2 = jieba.lcut(content, cut_all=True)
    print(f'全模式分词result2--》{result2}')


#todo: 03-jieba搜素引擎分词
def demo3_jieba():
    # 待切分的文本
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
    # result1 = jieba.cut_for_search(content) # generator object
    # for vaue in result1:
    #     print(vaue)
    result2 = jieba.lcut_for_search(content)
    print(f'搜索引擎分词result2--》{result2}')


#todo: 04-jieba支持繁体分词
def demo4_jieba():
    # 待切分的文本
    content = "煩惱即是菩提，我暫且不提"
    result2 = jieba.lcut(content) # 默认精确分词
    print(f'result2--》{result2}')

#todo: 05-jieba支持自定义词典：按照词典里面的词优先分词

def demo5_jieba():
    # 待切分的文本
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
    # 没有加入自定义词典的词
    result1 = jieba.lcut(content) # 默认精确分词
    print(f'result1--》{result1}')
    # 使用用户自定义词典
    jieba.load_userdict('./userdict.txt')
    result2 = jieba.lcut(content) # 默认精确分词
    print(f'result2--》{result2}')
if __name__ == '__main__':
    # demo1_jieba()
    # demo2_jieba()
    # demo3_jieba()
    # demo4_jieba()
    demo5_jieba()