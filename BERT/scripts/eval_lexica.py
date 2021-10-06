import pandas as pd
import numpy as np
import random
from tqdm import tqdm

import torch
import spacy

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

def getLexicon(lexicon):

    try:
        lexiconWords = set(lexicon.Word.values)
    except:
        lexiconWords = set(lexicon.word.values)

    lexiconMap = {}

    for i in range(len(lexicon)):
        lexiconMap[lexicon.iloc[i]['Word']] = lexicon.iloc[i]['Value']
        #lexiconMap[lexicon.iloc[i]['word']] = lexicon.iloc[i]['score']
        
    return lexiconWords, lexiconMap


def scoreText(text, lexiconWords, lexiconMap, tokenizer):
    
    score = 0
    
    tokens = [token.text.lower() for token in tokenizer(text)]
    #tokens = tokenizer.tokenize(text)
    
    if len(tokens) == 0:
        return None
    
    for token in tokens:
        if token in lexiconWords:
            score += lexiconMap[token]
            
    return score/len(tokens)


def evaluateLexicon(testDf, lexicon, tokenizer, task='classification'):
    
    print(tokenizer)
    lexiconWords, lexiconMap = getLexicon(lexicon)
    
    ### Getting lexicon scores for text
    scoreList = []

    for i in tqdm(range(len(testDf))):
        score = scoreText(testDf.iloc[i]['text'].lower(), lexiconWords, lexiconMap, tokenizer)

        scoreList.append(score)
        
    testDf['score'] = scoreList
    testDf.dropna(inplace=True)
    #print(testDf.shape)
    
    ### Training model for classification
    model = LogisticRegression()
    X = testDf.score.values.reshape(-1,1)
    y = testDf.label

    ### Computing Metrics
    acc = np.round(np.mean(cross_val_score(model, X, y, cv=5, scoring='accuracy')),3)
    f1 = np.round(np.mean(cross_val_score(model, X, y, cv=5, scoring='f1')),3)
    if task == 'classification':
        precision = np.round(np.mean(cross_val_score(model, X, y, cv=5, scoring='average_precision')),3)
        auc = np.round(np.mean(cross_val_score(model, X, y, cv=5, scoring='roc_auc')),3)
    elif task == 'regression':
        precision = np.round(np.mean(cross_val_score(model, X, y, cv=5, scoring='precision_micro')),3)
        auc = np.nan
        
    print(acc, f1)
    #return acc, f1


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

result = []
for i in ['nrc_joy', 'yelp_subset', 'empathy', 'amazon_finefood_subset', 'amazon_toys_subset',]:
#for i in ['nrc_joy']:
    for j in ['song_joy','emobank_V', 'dialog_joy', 'amazon_finefood_subset', 'yelp_subset', 'empathy', 'nrc_joy', 'amazon_toys_subset', 'friends_joy']:
        print(i,j)
        lexicon = pd.read_csv("empathy_dictionary/lexica/DistilBERT_Partition/"+i+"_distilbert_classification_ps.csv")
        df = getData("Raw", j, "classification")
        if j in ['yelp_subset', 'empathy', 'amazon_finefood_subset', 'amazon_toys_subset', ] + ['nrc_joy']:
            _, _, df = splitData(df, True)
        else:
            df = balanceData(df)
        evaluateLexicon(df, lexicon, tokenizer)
        #acc, f1 = evaluateLexicon(df, lexicon, tokenizer)
        #result.append([i,j, acc, f1])
#pd.DataFrame(result).to_csv("result.csv")