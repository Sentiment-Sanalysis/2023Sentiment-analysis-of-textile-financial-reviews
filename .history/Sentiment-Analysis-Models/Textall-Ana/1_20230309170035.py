from tkinter import *
import numpy as np
import jieba as jb
import joblib
from gensim.models.word2vec import Word2Vec

wv = Word2Vec.load("./data/")