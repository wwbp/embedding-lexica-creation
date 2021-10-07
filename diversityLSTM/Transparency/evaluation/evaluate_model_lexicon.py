import pickle
import argparse
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import spacy


parser = argparse.ArgumentParser()

parser.add_argument("--dataFolder", required=True, type=str, help="The input data dir.")
args = parser.parse_args()


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


def eval_lex(train_dataset_name, eval_dataset_name, nlp):
    path = 'diversityLSTM/Transparency/lexica/'
    filename = path + train_dataset_name + '_lstm_attention.csv'
    lexiconWords, lexiconMap = getLexicon(file = filename)
    if ('dialog' not in eval_dataset_name)and('song' not in eval_dataset_name)and('friends' not in eval_dataset_name)and('emobank' not in eval_dataset_name):
        path = os.path.join(args.dataFolder, eval_dataset_name)
        testDf = pd.read_csv(os.path.join(path, 'test.csv'))
    else:
        path = os.path.join(args.dataFolder, 'test_datasets')
        testDf = pd.read_csv(os.path.join(path, eval_dataset_name+'.csv'))

    lexAcc, lexF1 = evaluateLexicon(testDf, lexiconWords, lexiconMap, nlp, returnScores = True)
    return lexAcc, lexF1


def eval_model(X,y):
    ## Evaluating the 5-fold performance of the LogReg model 
    logModel = LogisticRegression()
    modelAcc = np.round(np.mean(cross_val_score(logModel, X, y, cv=5, scoring='accuracy')),3)
    modelF1 = np.round(np.mean(cross_val_score(logModel, X, y, cv=5, scoring='f1')),3)
    # precision = np.round(np.mean(cross_val_score(logModel, X, y, cv=5, scoring='average_precision')),3)
    # auc = np.round(np.mean(cross_val_score(logModel, X, y, cv=5, scoring='roc_auc')),3)
    return modelAcc, modelF1


def main():
    path = 'diversityLSTM/Transparency/evaluation/model_outputs/'
    train_dataset_list = ['nrc_joy','nrc_anger','nrc_sadness','nrc_surprise','nrc_fear','amazon_toys_subset','amazon_finefood_subset','yelp_subset']
    posneg_train_list = ['nrc_joy','amazon_toys_subset','amazon_finefood_subset','yelp_subset']
    posneg_eval_list = ['nrc_joy','friends_joy','song_joy','dialog_joy','emobank','amazon_toys_subset','amazon_finefood_subset','yelp_subset']
    nlp = spacy.load('./fasttext')

    results = []
    for train_dataset_name in train_dataset_list:
        if train_dataset_name in posneg_train_list:
            for eval_dataset_name in posneg_eval_list:
                X = pickle.load(open(path+train_dataset_name+'_'+eval_dataset_name+'_predictions_pdump.pkl','rb'))
                y = pickle.load(open(path+train_dataset_name+'_'+eval_dataset_name+'_true_label_pdump.pkl','rb'))
                modelAcc, modelF1 = eval_model(X,y)
                lexAcc, lexF1 = eval_lex(train_dataset_name, eval_dataset_name ,nlp)
                # print(f" {train_dataset_name}, {eval_dataset_name} , {modelAcc} , {modelF1}, {precision}, {auc}")
                print(f" {train_dataset_name}, {eval_dataset_name} , {modelAcc} , {modelF1}, {lexAcc}, {lexF1}")
                results.append([train_dataset_name, eval_dataset_name, modelAcc, modelF1, lexAcc, lexF1])
        else:
            emotion = train_dataset_name.split('_',1)[1]
            for eval_dataset in ['nrc_', 'song_', 'dialog_', 'friends_']:
                eval_dataset_name = eval_dataset + emotion
                X = pickle.load(open(path+train_dataset_name+'_'+eval_dataset_name+'_predictions_pdump.pkl','rb'))
                y = pickle.load(open(path+train_dataset_name+'_'+eval_dataset_name+'_true_label_pdump.pkl','rb'))
                modelAcc, modelF1 = eval_model(X,y)
                lexAcc, lexF1 = eval_lex(train_dataset_name, eval_dataset_name ,nlp)
                print(f" {train_dataset_name}, {eval_dataset_name} , {modelAcc} , {modelF1}, {lexAcc}, {lexF1}")
                results.append([train_dataset_name, eval_dataset_name, modelAcc, modelF1, lexAcc, lexF1])

    results = pd.DataFrame(results)
    results.columns = ["TrainData","TestData","modelAcc", "modelF1", "lexiconAcc", "lexiconF1"]
    results.to_csv("Results_LSTM.csv",index = False, index_label = False)

if __name__ == "__main__":
    main()

