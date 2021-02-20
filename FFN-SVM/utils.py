import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import os
import sys
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords 

from transformers import BertForSequenceClassification, BertTokenizer, BertForMaskedLM

from simpletransformers.language_modeling import LanguageModelingModel

from sklearn.metrics.pairwise import cosine_similarity, paired_euclidean_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler

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

from sklearn.feature_extraction.text import CountVectorizer

from typing import List
from pathlib import Path
from joblib import dump, load

import plotly.graph_objects as go

import itertools
from collections import Counter

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import LinearSVC

sys.path.insert(0,'/data2/Datasets/')
from preprocessing.preprocess import *






def generateFastTextData(trainDf, embedder, textVariable = 'text'):
    
    vecList = []
    
    for i in tqdm(range(len(trainDf))):
        vec = embedder(trainDf.iloc[i][textVariable].lower())
        vecList.append(vec.reshape(1,-1))
    
    finalData = np.concatenate(vecList)

    return finalData


       
def generateFastTextData_Spacy(trainDf, embedder, textVariable = 'text'):
    
    vecList = []
    
    for i in range(len(trainDf)):
        with embedder.disable_pipes():
            vec = embedder(trainDf.iloc[i][textVariable].lower()).vector
        vec = vec.reshape(1,-1)
        
        if vec.shape != (1,300):
            print("Error ")
            print(i)
        vecList.append(vec.reshape(1,-1))
    
    finalData = np.concatenate(vecList)

    return finalData



def getWordCount(df, nlp):

    tokenList = []

    for i in range(len(df)):

        doc = nlp(df.iloc[i]['text'].lower())
        words = [token.text for token in doc]

        tokenList.append(words)
        
    wordCount = list(itertools.chain.from_iterable(tokenList))
        
    word_freq = Counter(wordCount)
    
    df = pd.Series(word_freq).reset_index()
    df.columns = ['word','wordCount']
    
    return df


class Dataset(torch.utils.data.Dataset):
  def __init__(self, df, data):


    self.df = df
    self.data = data


  def __len__(self):

    return len(self.df)

  def __getitem__(self, index):
        
    feat = self.data[index]


    label = self.df.iloc[index]['label']

    return feat, label


    
class NNNet(nn.Module):
    def __init__(self):
        super(NNNet, self).__init__()
        
        self.fc1 = nn.Linear(300, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 2)
                

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
#         x = nn.Dropout()(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
    
###########################    
###### LEXICON EVALUATION CODES
###########################


def getLexicon(file=None, df = None):
    
    if df is None:
        lexicon = pd.read_csv(file)
    else:
        lexicon = df
    
    if 'scores' in lexicon.columns:
        lexicon.rename({'scores':'score'},axis =1, inplace = True)

    lexiconWords = set(lexicon.word.values)

    lexiconMap = {}

    for i in range(len(lexicon)):
        lexiconMap[lexicon.iloc[i]['word']] = lexicon.iloc[i]['score']
        
    return lexiconWords, lexiconMap


def scoreText(text, lexiconWords, lexiconMap, nlp):
    
    score = 0
    
    
    doc = nlp(text.lower())
    tokenList = [token.text for token in doc]
    
    counts = 0

    for token in tokenList:
        if token in lexiconWords:
            score += lexiconMap[token]
            counts += 1
           
    if counts==0:
        return 0.0
    
    return score/counts

 
def evaluateLexicon(testDf, lexiconWords, lexiconMap, nlp, dataName=None, lexiconName=None, returnScores = False):
    
    ### Getting lexicon scores for text
    scoreList = []

    for i in range(len(testDf)):
        score = scoreText(testDf.iloc[i]['text'].lower(), lexiconWords, lexiconMap, nlp)

        scoreList.append(score)
        
    testDf['score'] = scoreList
    
    ### Training model for classification
    model = LogisticRegression()
    X = testDf.score.values.reshape(-1,1)
    y = testDf.label
    
    
    ### Computing Metrics
    acc = np.round(np.mean(cross_val_score(model, X, y, cv=5, scoring='accuracy')),3)
    f1 = np.round(np.mean(cross_val_score(model, X, y, cv=5, scoring='f1')),3)
    precision = np.round(np.mean(cross_val_score(model, X, y, cv=5, scoring='average_precision')),3)
    auc = np.round(np.mean(cross_val_score(model, X, y, cv=5, scoring='roc_auc')),3)
    
    if returnScores:
        return acc, f1

#     print(f" ACC | F1 | Precision | AUC ")
    if dataName is None:
        print(f" {acc} , {f1} , {precision} , {auc}")
    else:
        print(f" {dataName} , {lexiconName} , {acc} , {f1} , {precision} , {auc}")
        
        
###########################    
###### SVM EVALUATION
###########################
def getWordPred_SVM(model, trainDf, nlp ):
    

    data = trainDf
        
    ## Getting the word count of all words
    wordCount = getWordCount(data, nlp)
    wordList = list(wordCount.word)
    
    ## Generating the embeddings for all words in the data
    embList = []
    for word in wordList:
        
        with nlp.disable_pipes():
            emb = nlp(word.lower()).vector.reshape(1,-1)
            embList.append(emb)

    embData = np.concatenate(embList,0)
    
    
    ## Scoring all words using the SVM model
    svmScores = model.decision_function(embData)


    wordDf = pd.DataFrame({'word':wordList,'score':svmScores})
    wordDf = wordDf.merge(wordCount, on='word')
    wordDf = wordDf.sort_values('score',ascending = False)
    wordDf['SVM_Rank'] = list(range(len(wordDf)))
   
    return wordDf


def trainSVM(trainData, testData, trainDf, testDf):
  
    model = LinearSVC(random_state=1, dual=False, max_iter=10000, C = 25.0) 
    model.fit(trainData, trainDf.label)  
    pred = model.predict(testData)
         
    return model
    
    
def generateLexicon_SVM(model, trainDf, nlp):
    return getWordPred_SVM(model, trainDf, nlp)

def testSVM( model, dataset, lexiconWords, lexiconMap, nlp, dataFolder, train = False):
    
    ##Loading the dataset
    if ('dialog' not in dataset) and ('song' not in dataset) and ('friends' not in dataset) and ('emobank' not in dataset):
        trainDf, devDf, testDf = splitData(getData(dataFolder, dataset))
    else:
        testDf = balanceData(getData(dataFolder, dataset))
        
    
    ## Generating the embedding data using Spacy     
    if train:
        testData = generateFastTextData_Spacy(trainDf, nlp, textVariable = 'text')
        testDf = trainDf
    else:
        testData = generateFastTextData_Spacy(testDf, nlp, textVariable = 'text')
        
    ## Generating the scores from the trained SVM model for the testing data
    scores = model.decision_function(testData)
    X = scores.reshape(-1,1)
    y = testDf.label.values
    
    ## Evaluating the 5-fold performance of the LogReg model 
    logModel = LogisticRegression()
    modelAcc = np.round(np.mean(cross_val_score(logModel, X, y, cv=5, scoring='accuracy')),3)
    modelF1 = np.round(np.mean(cross_val_score(logModel, X, y, cv=5, scoring='f1')),3)
    precision = np.round(np.mean(cross_val_score(logModel, X, y, cv=5, scoring='average_precision')),3)
    auc = np.round(np.mean(cross_val_score(logModel, X, y, cv=5, scoring='roc_auc')),3)
    
    ## Evaluating the performance of the Lexicon
    lexAcc, lexF1 = evaluateLexicon(testDf, lexiconWords, lexiconMap, nlp, returnScores = True)
    
    print(f"{dataset} , {modelAcc} , {modelF1} , {lexAcc} , {lexF1}")
    
    return dataset, modelAcc, modelF1, lexAcc, lexF1


###########################
####### FFN EVALUATION
###########################


def testModel_FFN(net, testLoader, returnScore = False, device='cuda:0'):
    net.eval()

    preds = []
    targets = []
    scores = []

    with torch.no_grad():

        for batch_idx, (data,target) in tqdm(enumerate(testLoader)):

            data, target = data.to(device), target.to(device)            
            output = net(data)
            
            scores.append(nn.Softmax()(output)[:,1].cpu().numpy())

            preds += np.argmax(output.cpu().numpy(),1).tolist()

            targets.append( target.cpu().numpy() )

    target =np.concatenate(targets)
    scores =np.concatenate(scores)
    
    acc = accuracy_score(preds, target)
    f1 = f1_score(preds,target)
    
    net.train()
    
    if returnScore:
        return acc, f1, scores
    else:
        return acc, f1
    
    
def getWordPred_FFN(NNnet, trainDf, nlp, device = 'cuda:0' ):
    

    NNnet.to(device)

    data = trainDf
        
    
    wordCount = getWordCount(data, nlp)
    wordList = list(wordCount.word)
    
    embList = []
    for word in wordList:
        
        with nlp.disable_pipes():
            emb = nlp(word.lower()).vector.reshape(1,-1)
            embList.append(emb)

    embData = np.concatenate(embList,0)
    
    
    ### NN
    
    NNnet.eval()
    wordPred = NNnet(torch.from_numpy(embData).to(device))

    NNwordPred = wordPred.detach().cpu().numpy()[:,1]
    NNwordDf = pd.DataFrame({'word':wordList,'NNprob':NNwordPred})
    NNwordDf = NNwordDf.merge(wordCount, on='word')
    NNwordDf = NNwordDf.sort_values('NNprob',ascending = False)
    NNwordDf['NN_Rank'] = list(range(len(NNwordDf)))
    
    
    return NNwordDf

    
def train_epoch_FFN(net, trainLoader, testLoader, optimizer, device='cuda:0'):
    
    net.train()
    
    classLoss = nn.CrossEntropyLoss()
    
    for batch_idx, (data,target) in enumerate(trainLoader):

        data, target = data.to(device), target.to(device)        

        optimizer.zero_grad()

        output = net(data)

        loss = classLoss(output, target)

        loss.backward()

        optimizer.step()


    acc,f1 = testModel_FFN(net, testLoader, device=device)
    print(f"Acc : {acc} F1 : {f1}")
    print("*"*25)

    
    
def trainFFN(trainData, testData, num_epochs = 5, batchSize = 5, device='cuda:0'):
    
        
    NNnet = NNNet()    
    NNnet.to(device)

    classLoss = nn.CrossEntropyLoss()
    reconLoss = nn.MSELoss()


    optimizer_NN = torch.optim.Adam(filter(lambda p: p.requires_grad, NNnet.parameters()), lr=0.0001)

    

    trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batchSize, 
                                              shuffle=True, num_workers=1)


    testLoader = torch.utils.data.DataLoader(testData, batch_size=batchSize, 
                                              shuffle=False, num_workers=1)
    
    
    for i in range(num_epochs):
        train_epoch_FFN(NNnet, trainLoader, testLoader, optimizer_NN, device)
        
        
    return NNnet
    
    
def generateLexicon_FFN(NNnet, trainDf, nlp, device = 'cuda:0'):
    
    wordPred = getWordPred_FFN(NNnet, trainDf, nlp, device)
    
    return wordPred


def testFFN( NNnet, dataset, lexiconWords, lexiconMap, nlp, dataFolder, train = False):
    
    if ('dialog' not in dataset) and ('song' not in dataset) and ('friends' not in dataset) and ('emobank' not in dataset):
        trainDf, devDf, testDf = splitData(getData(dataFolder, dataset))
    else:
        testDf = balanceData(getData(dataFolder, dataset))

        
    if train:
        testData = generateFastTextData_Spacy(trainDf, nlp, textVariable = 'text')
        testDataset = Dataset(trainDf, testData)
        testDf = trainDf
    else:
        testData = generateFastTextData_Spacy(testDf, nlp, textVariable = 'text')
        testDataset = Dataset(testDf, testData)
        
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=5, 
                                              shuffle=False, num_workers=1)
    
    
    modelAcc, modelF1, scores = testModel_FFN(NNnet, testLoader, returnScore = True)
    
    
    
    X = scores.reshape(-1,1)
    y = testDf.label.values
    
    ## Evaluating the 5-fold performance of the LogReg model 
    logModel = LogisticRegression()
    modelAcc = np.round(np.mean(cross_val_score(logModel, X, y, cv=5, scoring='accuracy')),3)
    modelF1 = np.round(np.mean(cross_val_score(logModel, X, y, cv=5, scoring='f1')),3)
    precision = np.round(np.mean(cross_val_score(logModel, X, y, cv=5, scoring='average_precision')),3)
    auc = np.round(np.mean(cross_val_score(logModel, X, y, cv=5, scoring='roc_auc')),3)
    
    
    
    lexAcc, lexF1 = evaluateLexicon(testDf, lexiconWords, lexiconMap, nlp, returnScores = True)
    
    print(f"{dataset} , {modelAcc} , {modelF1} , {lexAcc} , {lexF1}")
    
    return dataset, modelAcc, modelF1, lexAcc, lexF1
    
