import logging
from tqdm import tqdm
import itertools
from collections import Counter

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import torch
import torch.nn.functional as F
from torch import nn

from preprocessing.preprocess import *


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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
            logger.warning("Error ")
            logger.warning(i)
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
        self.drop = nn.Dropout(0.7)        
                
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        # x = self.drop(x)
        x = F.relu(self.fc2(x))
        # x = self.drop(x)
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

#     logger.info(f" ACC | F1 | Precision | AUC ")
    if dataName is None:
        logger.info(f" {acc} , {f1} , {precision} , {auc}")
    else:
        logger.info(f" {dataName} , {lexiconName} , {acc} , {f1} , {precision} , {auc}")
        
        
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


def trainSVM(trainData, trainDf,):
  
    model = LinearSVC(random_state=1, dual=False, max_iter=10000, C = 25.0) 
    model.fit(trainData, trainDf.label)  
         
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
    
    ## Evaluating the performance of the Lexicon
    lexAcc, lexF1 = evaluateLexicon(testDf, lexiconWords, lexiconMap, nlp, returnScores = True)
    
    logger.info(f"{dataset} , {modelAcc} , {modelF1} , {lexAcc} , {lexF1}")
    
    return [dataset, modelAcc, modelF1, lexAcc, lexF1]


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

    
def trainFFN(trainData, testData, max_epochs = 5, batchSize = 5, device='cuda:0'):
    
        
    NNnet = NNNet()    
    NNnet.to(device)

    optimizer_NN = torch.optim.Adam(filter(lambda p: p.requires_grad, NNnet.parameters()), lr=0.0001, weight_decay=1e-3)

    

    trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batchSize, 
                                              shuffle=True, num_workers=1)


    testLoader = torch.utils.data.DataLoader(testData, batch_size=batchSize, 
                                              shuffle=False, num_workers=1)
    
    
    NNnet.train()
    classLoss = nn.CrossEntropyLoss()
    best_f1 = 0
    tol = 0

    for _ in range(max_epochs):
        for batch_idx, (data,target) in enumerate(trainLoader):

            data, target = data.to(device), target.to(device)        

            optimizer_NN.zero_grad()

            output = NNnet(data)

            loss = classLoss(output, target)

            loss.backward()

            optimizer_NN.step()
        
        acc,f1 = testModel_FFN(NNnet, trainLoader, device=device)
        logger.info(f"Train Acc : {acc} Train F1 : {f1}")
        
        acc,f1 = testModel_FFN(NNnet, testLoader, device=device)
        logger.info(f"Validation Acc : {acc} Validation F1 : {f1}")

        if f1 > best_f1:
            best_f1 = f1
            tol = 0
            best_model = NNnet
            continue
        else:
            tol += 1

        if tol >= 5:
            break
        
    return best_model
    
    
def generateLexicon_FFN(NNnet, trainDf, nlp, device = 'cuda:0'):
    
    wordPred = getWordPred_FFN(NNnet, trainDf, nlp, device)
    
    return wordPred


def testFFN( NNnet, dataset, lexiconWords, lexiconMap, nlp, dataFolder, device, train = False):
    
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
    
    modelAcc, modelF1, scores = testModel_FFN(NNnet, testLoader, returnScore = True, device=device)
    
    X = scores.reshape(-1,1)
    y = testDf.label.values
    
    ## Evaluating the 5-fold performance of the LogReg model 
    logModel = LogisticRegression()
    modelAcc = np.round(np.mean(cross_val_score(logModel, X, y, cv=5, scoring='accuracy')),3)
    modelF1 = np.round(np.mean(cross_val_score(logModel, X, y, cv=5, scoring='f1')),3)

    lexAcc, lexF1 = evaluateLexicon(testDf, lexiconWords, lexiconMap, nlp, returnScores = True)
    
    logger.info(f"{dataset} , {modelAcc} , {modelF1} , {lexAcc} , {lexF1}")
    
    return [dataset, modelAcc, modelF1, lexAcc, lexF1]
    
