# -*- encoding: utf-8 -*-
'''
@File    :   Wordcloud.py-》
@Time    :   2023/03/05 22:55:38
@Author  :   ZihanWang
'''
'''
    Created by Asen on 2023/3/5
'''

import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def word_cloud(text,outpass):

    # 使用jieba对文本进行分词
    words = jieba.cut(text)

    # 将分词结果转换为空格分隔的字符串
    words_str = ' '.join(words)

    #停词
    std = open('./stopwords.txt',encoding="utf-8").read()

    # 用户自建停词表加入
    # std += newsw

    # 生成词云
    wc = WordCloud(
        # 参数列表
        font_path='msyh.ttc',  # 设置字体
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
    # 也可以使用plt.savefig()保存图片
    plt.savefig(outpass)
    return plt
    # plt.show()


if __name__ == '__main__':
    filepath = './test.txt'
    plt = word_cloud(filepath,'./wc.png')
    plt.show()

from typing import Union
# 引入fastapi包
from fastapi import FastAPI,File, UploadFile

# 引入设置预设值所需的Enum类
from enum import Enum

# 使用 Pydantic 模型来声明请求体，并能够获得它们所具有的所有能力和优点。
from pydantic import BaseModel

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


# 创建一个FastAPI实例
app = FastAPI()

# 定义一个路径操作装饰器


@app.post("/")
# 路径：是 /。
# 操作：是 get。
# 函数：是位于「装饰器」下方的函数（位于 @app.get("/") 下方）。
# 路径操作函数
async def root():
    return {"message": "Hello World"}

# 你可以使用标准的 Python 类型标注为函数中的路径参数声明类型。

# 类型检查：int


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    # 可以返回一个 dict、list，像 str、int 一样的单个值，等等
    return {"item_id": item_id}


# 预制值
@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}


# 接收文件与表单字段
@app.post("/files/")
async def create_file(file: Union[bytes, None] = File(default=None)):
    if not file:
        return {"message": "No file sent"}
    else:
        return { word_cloud(file,'./wc.png')}

# 如果把路径操作函数参数的类型声明为 bytes，FastAPI 将以 bytes 形式读取和接收文件内容。

# 这种方式把文件的所有内容都存储在内存里，适用于小型文件。

# 不过，很多情况下，UploadFile 更好用。
@app.post("/uploadfile/")
async def create_upload_file(
    file: UploadFile = File(description="A file read as UploadFile"),
):
    return {"filename": file.filename}