import pandas as pd
import numpy as np
import seaborn as sns
import os

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords 


from sklearn.metrics.pairwise import cosine_similarity, paired_euclidean_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import itertools
from collections import Counter

from tqdm import tqdm
import torch



import matplotlib.pyplot as plt


from functools import partial

import pickle

from collections import deque


import torch.nn.functional as F
from torch import nn

from sklearn.utils import resample

import fasttext




from typing import List
from pathlib import Path
from joblib import dump, load



import spacy
import warnings
warnings.filterwarnings("ignore")
import sys


######### USER INPUTS ###########

sys.path.insert(0,'/data1/YelpAnalysis/')
from utils import *


nlp = spacy.load('/data2/link10/models/fasttext/en_fasttext_crawl')

sys.path.insert(0,'/data2/Datasets/')
from preprocess import *

dataFolder = '/data2/Datasets/Raw'

## Dataset that will be used for creating the lexicon
lexiconDataset = 'nrc_joy'
dataList = ['nrc_joy','song_joy','dialog_joy','friends_joy']

device = 'cuda:0'

##################################




trainDf, devDf, testDf = splitData(getData(dataFolder, lexiconDataset))
trainData = generateFastTextData_Spacy(trainDf, nlp, textVariable = 'text')

testData = generateFastTextData_Spacy(testDf, nlp, textVariable = 'text')

trainDataset = Dataset(trainDf, trainData)
testDataset = Dataset(testDf, testData)



NNnet = trainFFN(trainDataset, testDataset, num_epochs = 3)


lexicon = generateLexicon_FFN(NNnet,trainDf, nlp, device =device)

lexicon.rename({'NNprob':'score'},axis = 1, inplace = True)
lexiconWords, lexiconMap = getLexicon(df = lexicon)

results = []

for data in dataList:
    results.append(testFFN(NNnet,data,lexiconWords, lexiconMap, nlp, dataFolder))
    
results = pd.DataFrame(results)
results.columns = ['TestData','modelAcc', 'modelF1', 'lexiconAcc', 'lexiconF1']
results.to_csv("Results.csv",index = False, index_label = False)

























