import pandas as pd
import numpy as np
import seaborn as sns
import os

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords 

from transformers import BertForSequenceClassification, BertTokenizer, BertForMaskedLM

from simpletransformers.language_modeling import LanguageModelingModel

from sklearn.metrics.pairwise import cosine_similarity, paired_euclidean_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import itertools
from collections import Counter
import sys
from tqdm import tqdm
import torch

import networkx as nx

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from functools import partial

import pickle

from collections import deque

from torchvision import models
import torch.nn.functional as F
from torch import nn

from sklearn.utils import resample

import fasttext
import sister


import spacy


from sklearn.svm import LinearSVC
from typing import List
from pathlib import Path
from joblib import dump, load

import plotly.graph_objects as go


######### USER INPUTS ###########

sys.path.insert(0,'/data1/YelpAnalysis/')
from utils import *

nlp = spacy.load('/data2/link10/models/fasttext/en_fasttext_crawl')

sys.path.insert(0,'/data2/Datasets/')
from preprocess import *

dataFolder = '/data2/Datasets/Raw'
lexiconFolder = '/data1/YelpAnalysis/ExternalData_Evaluation/Model_vs_Lexicon/SVM/Lexicons/'


dataList = ['nrc_joy', 'yelp_subset','amazon_finefood_subset','amazon_toys_subset','empathy']

##################################


for data in dataList:
    trainDf, devDf, testDf = splitData(getData(dataFolder, data))
    trainData = generateFastTextData_Spacy(trainDf, nlp, textVariable = 'text')
    testData = generateFastTextData_Spacy(testDf, nlp, textVariable = 'text')

    model = trainSVM(trainData, testData, trainDf, testDf)

    lexiconDf = generateLexicon_SVM(model,trainDf, nlp)
    
    outfilename = f"{lexiconFolder}/{data}_lexicon.csv"
    
    lexiconDf.to_csv(outfilename, index = False, index_label = False)


