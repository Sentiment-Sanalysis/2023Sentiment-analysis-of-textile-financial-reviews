# -*- encoding: utf-8 -*-
'''
@File    :   app.py
@Time    :   2023/03/05 22:55:38
@Author  :   ZihanWang
'''

import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def word_cloud(filepath):

    # 打开文本文件，读取内容
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # 使用jieba对文本进行分词
    words = jieba.cut(text)

    # 将分词结果转换为空格分隔的字符串
    words_str = ' '.join(words)

    #停词
    std = open('./stopwords.txt',encoding="utf-8").read()

    # 生成词云
    wc = WordCloud(
        # 参数列表
        font_path='HGKT_CNKI.TTF',  # 设置字体
        background_color='white',  # 设置背景颜色
        width=800,  # 设置宽度
        height=600,  # 设置高度
        max_words=200,  # 设置最大词数
        max_font_size=100,  # 设置最大字体大小
        stopwords=std # 设置停用词
    )

    # 生成词云图片
    wc.generate(words_str)
    # 显示词云图片
    plt.imshow(wc)
    plt.axis('off')
    # 返回词云图片plt对象,module类型
    # 可以使用plt.show()显示图片
    plt.savefig('./wc.png')
    return plt
    # plt.show()

if __name__ == '__main__':
    filepath = './test.txt'
    plt = word_cloud(filepath)
    plt.show()
