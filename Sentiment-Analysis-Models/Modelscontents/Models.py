# -*- encoding: utf-8 -*-
'''
@File    :   Models.py
@Time    :   2023/03/10 20:11:53
@Author  :   ZihanWang 
'''
import pickle
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from pandas import read_csv
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import jieba


def Models(filepath,model_name):
    """
    filepath: the path of the data
    model_name: the name of the model you want to use
    模型有：BRBM，LR，SVM，RF，KNN，NB，MLP，Bagging，AdaBoost，GBDT
    输入文件格式要求，两列，一列属性值为content，content为文本内容，
    另一列属性值为label，label为0或1
    函数输出为模型的准确率auc，召回率recall，精确率precision，F1值
    """
    # 读取数据
    data = pd.read_csv(filepath,encoding='utf-8')
    # 对content去停用词
    # 对所有数据分词
    words= []
    for i,row in  data.iterrows():
        word = jieba.cut(row['content'])
        result = '  '.join(word)
        words.append(result)
    # 特征向量化
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(words)
    X = X.toarray()
    # 转换成DataFrame
    X = pd.DataFrame(X)
    # 提取目标变量
    y = data['label']
    ## 划分训练集和测试集 ##
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 1)
    # 1.玻尔兹曼机
    if model_name == 'BRBM':
        from sklearn.neural_network import BernoulliRBM
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn import metrics
        # 构建模型
        model = Pipeline(steps=[('rbm', BernoulliRBM(n_components=100,learning_rate=0.01, n_iter=10, random_state=0)),
                                ('logistic', LogisticRegression())])
        # 训练模型
        model.fit(X_train, y_train)
        # 预测
        y_pred = model.predict(X_test)
        # 模型保存
        with open('BRBM.pkl', 'wb') as f:
            pickle.dump(model, f)
        # 模型评估
        return metrics.accuracy_score(y_test,y_pred),metrics.precision_score(y_test,y_pred),metrics.recall_score(y_test,y_pred),metrics.f1_score(y_test,y_pred)
    # 2.逻辑回归
    elif model_name == 'LR':
        from sklearn.linear_model import LogisticRegression
        from sklearn import metrics
        # 构建模型
        model = LogisticRegression()
        # 训练模型
        model.fit(X_train,y_train)
        # 预测
        y_pred = model.predict(X_test)
        # 模型保存
        with open('LR.pkl', 'wb') as f:
            pickle.dump(model, f)
        # 模型评估
        return metrics.accuracy_score(y_test,y_pred),metrics.precision_score(y_test,y_pred),metrics.recall_score(y_test,y_pred),metrics.f1_score(y_test,y_pred)

    # 3.支持向量机
    elif model_name == 'SVM':
        from sklearn import svm
        from sklearn import metrics
        # 构建模型
        model = svm.SVC()
        # 训练模型
        model.fit(X_train,y_train)
        # 预测
        y_pred = model.predict(X_test)
        # 模型保存
        with open('SVM.pkl', 'wb') as f:
            pickle.dump(model, f)
        # 模型评估
        return metrics.accuracy_score(y_test,y_pred),metrics.precision_score(y_test,y_pred),metrics.recall_score(y_test,y_pred),metrics.f1_score(y_test,y_pred)

    # 4.随机森林
    elif model_name == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        from sklearn import metrics
        # 构建模型
        model = RandomForestClassifier()
        # 训练模型
        model.fit(X_train,y_train)
        # 预测
        y_pred = model.predict(X_test)
        # 模型保存
        with open('RF.pkl', 'wb') as f:
            pickle.dump(model, f)
        # 模型评估
        return metrics.accuracy_score(y_test,y_pred),metrics.precision_score(y_test,y_pred),metrics.recall_score(y_test,y_pred),metrics.f1_score(y_test,y_pred)

    # 5.K近邻
    elif model_name == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn import metrics
        # 构建模型
        model = KNeighborsClassifier()
        # 训练模型
        model.fit(X_train,y_train)
        # 预测
        y_pred = model.predict(X_test)
        # 模型保存
        with open('KNN.pkl', 'wb') as f:
            pickle.dump(model, f)
        # 模型评估
        return metrics.accuracy_score(y_test,y_pred),metrics.precision_score(y_test,y_pred),metrics.recall_score(y_test,y_pred),metrics.f1_score(y_test,y_pred)

    # 6.朴素贝叶斯
    elif model_name == 'NB':
        from sklearn.naive_bayes import GaussianNB
        from sklearn import metrics
        # 构建模型
        model = GaussianNB()
        # 训练模型
        model.fit(X_train,y_train)
        # 预测
        y_pred = model.predict(X_test)
        # 模型保存
        with open('NB.pkl', 'wb') as f:
            pickle.dump(model, f)
        # 模型评估
        return metrics.accuracy_score(y_test,y_pred),metrics.precision_score(y_test,y_pred),metrics.recall_score(y_test,y_pred),metrics.f1_score(y_test,y_pred)

    # 7.决策树
    elif model_name == 'DT':
        from sklearn.tree import DecisionTreeClassifier
        from sklearn import metrics
        # 构建模型
        model = DecisionTreeClassifier()
        # 训练模型
        model.fit(X_train,y_train)
        # 预测
        y_pred = model.predict(X_test)
        # 模型保存
        with open('DT.pkl', 'wb') as f:
            pickle.dump(model, f)
        # 模型评估
        return metrics.accuracy_score(y_test,y_pred),metrics.precision_score(y_test,y_pred),metrics.recall_score(y_test,y_pred),metrics.f1_score(y_test,y_pred)

    # 8.神经网络
    elif model_name == 'NN':
        from sklearn.neural_network import MLPClassifier
        from sklearn import metrics
        # 构建模型
        model = MLPClassifier()
        # 训练模型
        model.fit(X_train,y_train)
        # 预测
        y_pred = model.predict(X_test)
        # 模型保存
        with open('NN.pkl', 'wb') as f:
            pickle.dump(model, f)
        # 模型评估
        return metrics.accuracy_score(y_test,y_pred),metrics.precision_score(y_test,y_pred),metrics.recall_score(y_test,y_pred),metrics.f1_score(y_test,y_pred)

    # 9.梯度提升决策树
    elif model_name == 'GBDT':
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn import metrics
        # 构建模型
        model = GradientBoostingClassifier()
        # 训练模型
        model.fit(X_train,y_train)
        # 预测
        y_pred = model.predict(X_test)
        # 模型保存
        with open('GBDT.pkl', 'wb') as f:
            pickle.dump(model, f)
        # 模型评估
        return metrics.accuracy_score(y_test,y_pred),metrics.precision_score(y_test,y_pred),metrics.recall_score(y_test,y_pred),metrics.f1_score(y_test,y_pred)

    # 10.AdaBoost
    elif model_name == 'AdaBoost':
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn import metrics
        # 构建模型
        model = AdaBoostClassifier()
        # 训练模型
        model.fit(X_train,y_train)
        # 预测
        y_pred = model.predict(X_test)
        # 模型保存
        with open('AdaBoost.pkl', 'wb') as f:
            pickle.dump(model, f)
        # 模型评估
        return metrics.accuracy_score(y_test,y_pred),metrics.precision_score(y_test,y_pred),metrics.recall_score(y_test,y_pred),metrics.f1_score(y_test,y_pred)

    # 11.Bagging
    elif model_name == 'Bagging':
        from sklearn.ensemble import BaggingClassifier
        from sklearn import metrics
        # 构建模型
        model = BaggingClassifier()
        # 训练模型
        model.fit(X_train,y_train)
        # 预测
        y_pred = model.predict(X_test)
        # 模型保存
        with open('Bagging.pkl', 'wb') as f:
            pickle.dump(model, f)
        # 模型评估
        return metrics.accuracy_score(y_test,y_pred),metrics.precision_score(y_test,y_pred),metrics.recall_score(y_test,y_pred),metrics.f1_score(y_test,y_pred)



    
    


