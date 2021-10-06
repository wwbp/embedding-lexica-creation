import os
import random
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import spacy


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    lexiconWords, lexiconMap = getLexicon(lexicon)
    
    ### Getting lexicon scores for text
    scoreList = []

    for i in range(len(testDf)):
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
        
    return [acc, f1]


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    tokenizer = spacy.load("./fasttext")

    lexFolder = './lexica/'
    dataFolder = './cleandata'
    methods = {'DistilBERT_Mask':'distilbert_classification_mask',
               'DistilBERT_Partition':'distilbert_classification_ps', 
               'Roberta_Mask':'roberta_classification_mask', 
               'Roberta_Partition':'roberta_classification_ps'}
    train_data1 = ["nrc_joy", "yelp_subset","amazon_finefood_subset","amazon_toys_subset",]
    train_data2 = ["surprise", "sadness", "fear", "anger"]

    test_data1 = train_data1+["song_joy", "dialog_joy", "friends_joy", "emobank"]
    test_data2 = ["nrc","song","dialog","friends"]

    results = [] 
    for method in methods:
        for i in train_data1:
            logger.info('Starting evaluating {} {}'.format(method, i))
            for j in test_data1:
                lexicon = pd.read_csv(lexFolder+method+'/'+i+'_'+methods[method]+'.csv')
                if ('dialog' not in j) and ('song' not in j) and ('friends' not in j) and ('emobank' not in j):
                    path = os.path.join(dataFolder, j)
                    df = pd.read_csv(os.path.join(path, 'test.csv'))
                else:
                    path = os.path.join(dataFolder, 'test_datasets')
                    df = pd.read_csv(os.path.join(path, j+'.csv'))
                results.append([method, i, j]+evaluateLexicon(df, lexicon, tokenizer))
        
        for i in train_data2:
            logger.info('Starting evaluating {} {}'.format(method, i))
            for j in test_data2:
                lexicon = pd.read_csv(lexFolder+method+'/'+'nrc_'+i+'_'+methods[method]+'.csv')
                if j == 'nrc':
                    path = os.path.join(dataFolder, j+'_'+i)
                    df = pd.read_csv(os.path.join(path, 'test.csv'))
                else:
                    path = os.path.join(dataFolder, 'test_datasets')
                    df = pd.read_csv(os.path.join(path, j+'_'+i+'.csv'))
                results.append([method, 'nrc_'+i, j+'_'+i]+evaluateLexicon(df, lexicon, tokenizer))
    
    results = pd.DataFrame(results)
    results.columns = ["Method","TrainData","TestData","lexiconAcc", "lexiconF1"]
    results.to_csv("Results_BERT.csv",index = False, index_label = False)


if __name__ == "__main__":
    main()