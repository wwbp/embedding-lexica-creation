from typing import *
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def getData(dataFolder: str, dataset: str, task: str) -> pd.DataFrame:
    """
    Available options for dataset
        -----------------------
        empathy
        distress
        -----------------------
        yelp
        yelp_subset
        -----------------------
        amazon_finefood
        amazon_finefood_subset
        amazon_toys
        amazon_toys_subset
        -----------------------
        nrc_joy
        nrc_sadness
        nrc_fear
        nrc_anger
        nrc_surprise
        -----------------------
        song_joy
        song_sadness
        song_fear
        song_anger
        song_surprise
        song_disgust
        -----------------------
        dialog_joy
        dialog_sadness
        dialog_fear
        dialog_anger
        dialog_surprise
        dialog_disgust
        -----------------------
        friends_joy
        friends_sadness
        friends_fear
        friends_anger
        friends_surprise
        friends_disgust
        -----------------------
        emobank_V
        emobank_A
        emobank_D
    """
#-----------------------------------------------------------------------------------------------
    
    if dataset in ['empathy', 'distress']:
        
        df = pd.read_csv(os.path.join(dataFolder,'Empathy/messages.csv'))
        df = df.dropna()
        if task == 'classification':
            df['label'] = df[dataset+'_bin']
        elif task == 'regression':
            df['label'] = df[dataset]
        df['text'] = df.essay
        df = df.filter(['text','label'], axis = 1)
        
#-----------------------------------------------------------------------------------------------
    
    elif dataset == 'yelp':

        df = pd.read_csv(os.path.join(dataFolder,'Yelp/df1M.tsv'), sep='\t')
        df = df.dropna()
        if task == 'classification':
            df = df[df.stars!=3]
            df['label'] = df.stars.apply(lambda x : 0 if x<3 else 1)
        elif task == 'regression':
            df['label'] = df.stars
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'yelp_subset':
        
        df = pd.read_csv(os.path.join(dataFolder,'Yelp/df1M.tsv'), sep='\t')
        df = df.dropna()
        if task == 'classification':
            df = df[df.stars!=3]
            df['label'] = df.stars.apply(lambda x : 0 if x<3 else 1)
        elif task == 'regression':
            df['label'] = df.stars
        df = df.filter(['text','label'], axis = 1)
        df = df.iloc[0:10000]
    
#-----------------------------------------------------------------------------------------------
    
    elif dataset == 'amazon_finefood':
        
        df = pd.read_csv(os.path.join(dataFolder, 'AmazonFineFood/Reviews.csv'))
        df = df.dropna()
        if task == 'classification':
            df = df[df.Score!=3]
            df['label'] = df.Score.apply(lambda x : 0 if x<3 else 1)
        elif task == 'regression':
            df['label'] = df.Score
        df['text'] = df.Text
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'amazon_finefood_subset':
        
        df = pd.read_csv(os.path.join(dataFolder, 'AmazonFineFood/Reviews.csv'))
        df = df.dropna()
        if task == 'classification':
            df = df[df.Score!=3]
            df['label'] = df.Score.apply(lambda x : 0 if x<3 else 1)
        elif task == 'regression':
            df['label'] = df.Score
        df['text'] = df.Text
        df = df.filter(['text','label'], axis = 1)  
        df = df.iloc[0:10000]  

    elif dataset == 'amazon_toys':
        
        df = pd.read_json(os.path.join(dataFolder, 'AmazonProductData/reviews_Toys_and_Games_5.json'), lines = True)
        df = df.dropna()
        if task == 'classification':
            df = df[df.overall!=3]
            df['label'] = df.overall.apply(lambda x : 0 if x<3 else 1)
        elif task == 'regression':
            df['label'] = df.overall
        df['text'] = df.reviewText
        df = df.filter(['text','label'], axis = 1)  

        
    elif dataset == 'amazon_toys_subset':
        
        df = pd.read_json(os.path.join(dataFolder, 'AmazonProductData/reviews_Toys_and_Games_5.json'), lines = True)
        df = df.dropna()
        if task == 'classification':
            df = df[df.overall!=3]
            df['label'] = df.overall.apply(lambda x : 0 if x<3 else 1)
        elif task == 'regression':
            df['label'] = df.overall
        df['text'] = df.reviewText
        df = df.filter(['text','label'], axis = 1)  
        df = df.iloc[0:10000]
        
#-----------------------------------------------------------------------------------------------
        
    elif dataset == 'nrc_joy':
        
        df = pd.read_csv(os.path.join(dataFolder,'NRCData/msgs_tec_langClean.csv'))
        df = df.dropna()
        df['label'] = df.emotion.apply(lambda x : 1 if x == 'joy' else 0)
        df['text'] = df.message
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'nrc_sadness':
        
        df = pd.read_csv(os.path.join(dataFolder,'NRCData/msgs_tec_langClean.csv'))
        df = df.dropna()
        df['label'] = df.emotion.apply(lambda x : 1 if x == 'sadness' else 0)
        df['text'] = df.message
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'nrc_fear':
        
        df = pd.read_csv(os.path.join(dataFolder,'NRCData/msgs_tec_langClean.csv'))
        df = df.dropna()
        df['label'] = df.emotion.apply(lambda x : 1 if x == 'fear' else 0)
        df['text'] = df.message
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'nrc_anger':
        
        df = pd.read_csv(os.path.join(dataFolder,'NRCData/msgs_tec_langClean.csv'))
        df = df.dropna()
        df['label'] = df.emotion.apply(lambda x : 1 if x == 'anger' else 0)
        df['text'] = df.message
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'nrc_surprise':
        
        df = pd.read_csv(os.path.join(dataFolder,'NRCData/msgs_tec_langClean.csv'))
        df = df.dropna()
        df['label'] = df.emotion.apply(lambda x : 1 if x == 'surprise' else 0)
        df['text'] = df.message
        df = df.filter(['text','label'], axis = 1)
        
#-----------------------------------------------------------------------------------------------

    elif dataset == 'song_joy':
        
        df = pd.read_csv(os.path.join(dataFolder,'SongLyrics/RadaEmotionWheelVersesCleaned.csv'))
        df = df.dropna()
        df['label'] = df.joy.apply(lambda x : 1 if x == 1.0 else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'song_sadness':
        
        df = pd.read_csv(os.path.join(dataFolder,'SongLyrics/RadaEmotionWheelVersesCleaned.csv'))
        df = df.dropna()
        df['label'] = df.sadness.apply(lambda x : 1 if x == 1.0 else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'song_fear':
        
        df = pd.read_csv(os.path.join(dataFolder,'SongLyrics/RadaEmotionWheelVersesCleaned.csv'))
        df = df.dropna()
        df['label'] = df.fear.apply(lambda x : 1 if x == 1.0 else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'song_anger':
        
        df = pd.read_csv(os.path.join(dataFolder,'SongLyrics/RadaEmotionWheelVersesCleaned.csv'))
        df = df.dropna()
        df['label'] = df.anger.apply(lambda x : 1 if x == 1.0 else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'song_surprise':
        
        df = pd.read_csv(os.path.join(dataFolder,'SongLyrics/RadaEmotionWheelVersesCleaned.csv'))
        df = df.dropna()
        df['label'] = df.surprise.apply(lambda x : 1 if x == 1.0 else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'song_disgust':
        
        df = pd.read_csv(os.path.join(dataFolder,'SongLyrics/RadaEmotionWheelVersesCleaned.csv'))
        df = df.dropna()
        df['label'] = df.disgust.apply(lambda x : 1 if x == 1.0 else 0)
        df = df.filter(['text','label'], axis = 1)
        
#-----------------------------------------------------------------------------------------------
        
    elif dataset == 'dialog_joy':
        
        df = pd.read_csv(os.path.join(dataFolder,'DailyDialog/dialogData.csv'), sep='\t')
        df = df.dropna()
        df['label'] = df.label.apply(lambda x : 1 if x == 'joy' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'dialog_sadness':
        
        df = pd.read_csv(os.path.join(dataFolder,'DailyDialog/dialogData.csv'), sep='\t')
        df = df.dropna()
        df['label'] = df.label.apply(lambda x : 1 if x == 'sadness' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'dialog_fear':
        
        df = pd.read_csv(os.path.join(dataFolder,'DailyDialog/dialogData.csv'), sep='\t')
        df = df.dropna()
        df['label'] = df.label.apply(lambda x : 1 if x == 'fear' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'dialog_anger':
        
        df = pd.read_csv(os.path.join(dataFolder,'DailyDialog/dialogData.csv'), sep='\t')
        df = df.dropna()
        df['label'] = df.label.apply(lambda x : 1 if x == 'anger' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'dialog_surprise':
        
        df = pd.read_csv(os.path.join(dataFolder,'DailyDialog/dialogData.csv'), sep='\t')
        df = df.dropna()
        df['label'] = df.label.apply(lambda x : 1 if x == 'surprise' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'dialog_disgust':
        
        df = pd.read_csv(os.path.join(dataFolder,'DailyDialog/dialogData.csv'), sep='\t')
        df = df.dropna()
        df['label'] = df.label.apply(lambda x : 1 if x == 'disgust' else 0)
        df = df.filter(['text','label'], axis = 1)
        
#-----------------------------------------------------------------------------------------------

    elif dataset == 'friends_joy':
        
        df = pd.read_csv(os.path.join(dataFolder,'EmotionLines/friendsData.csv'), sep='\t')
        df = df.dropna()
        df['label'] = df.label.apply(lambda x : 1 if x == 'joy' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'friends_sadness':
        
        df = pd.read_csv(os.path.join(dataFolder,'EmotionLines/friendsData.csv'), sep='\t')
        df = df.dropna()
        df['label'] = df.label.apply(lambda x : 1 if x == 'sadness' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'friends_fear':
        
        df = pd.read_csv(os.path.join(dataFolder,'EmotionLines/friendsData.csv'), sep='\t')
        df = df.dropna()
        df['label'] = df.label.apply(lambda x : 1 if x == 'fear' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'friends_anger':
        
        df = pd.read_csv(os.path.join(dataFolder,'EmotionLines/friendsData.csv'), sep='\t')
        df = df.dropna()
        df['label'] = df.label.apply(lambda x : 1 if x == 'anger' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'friends_surprise':
        
        df = pd.read_csv(os.path.join(dataFolder,'EmotionLines/friendsData.csv'), sep='\t')
        df = df.dropna()
        df['label'] = df.label.apply(lambda x : 1 if x == 'surprise' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'friends_disgust':
        
        df = pd.read_csv(os.path.join(dataFolder,'EmotionLines/friendsData.csv'), sep='\t')
        df = df.dropna()
        df['label'] = df.label.apply(lambda x : 1 if x == 'disgust' else 0)
        df = df.filter(['text','label'], axis = 1)
        
#-----------------------------------------------------------------------------------------------
        
    elif dataset in ['emobank_V', 'emobank_A', 'emobank_D']:
        
        df = pd.read_csv(os.path.join(dataFolder,'Empathy/emobank.csv'))
        df = df.dropna()
        if task == 'classification':
            df = df[df[dataset.split('_')[-1]]!=3]
            df['label'] = df[dataset.split('_')[-1]].apply(lambda x : 0 if x<3 else 1)
        elif task == 'regression':
            df['label'] = df[dataset.split('_')[-1]]
        df['text'] = df.essay
        df = df.filter(['text','label'], axis = 1)

    return df
    
    
def splitData(df: pd.DataFrame, balanceTrain: bool=True) -> Tuple[pd.DataFrame]:
    trainX, tempX, _, _ = train_test_split(df, df.label, test_size = 0.2, random_state = 42)
    devX, testX, _, _ = train_test_split(tempX, tempX.label, test_size = 0.5, random_state = 42)
    if balanceTrain:
        trainX = balanceData(trainX)
    return trainX, devX, testX   


def balanceData(df: pd.DataFrame) -> pd.DataFrame:
    ros = RandomOverSampler(random_state = 42)
    df, _ = ros.fit_resample(df, df.label)
    return df