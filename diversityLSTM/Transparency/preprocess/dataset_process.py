import pandas as pd 
import argparse
from preprocessing.preprocess import *

parser = argparse.ArgumentParser(description='Run Preprocessing on dataset')
parser.add_argument('--data_folder', type=str, required=True)
args = parser.parse_args()

dataFolder = args.data_folder
train_dataset_list = ['nrc_joy','nrc_anger','nrc_sadness','nrc_surprise','nrc_fear','empathy','amazon_toys_subset','amazon_finefood_subset','yelp_subset']
for dataset in train_dataset_list:
    train, dev, test = splitData(getData(dataFolder,dataset)) 
    train.loc[:,'exp_split'] = 'train'
    test.loc[:,'exp_split'] = 'test'
    dev.loc[:,'exp_split'] = 'dev'
    df = pd.concat([train,test,dev])
    filename = 'diversityLSTM/Transparency/preprocess/ourData/'+dataset + '_dataset.csv'
    df.to_csv(filename, index=False)
    print('{} loaded'.format(dataset))
external_dataset_list = ['friends_joy','friends_anger','friends_sadness','friends_surprise','friends_fear','dialog_joy','dialog_anger','dialog_sadness','dialog_surprise','dialog_fear','song_joy','song_anger','song_sadness','song_surprise','song_fear','emobank']
for dataset in external_dataset_list:
    df = balanceData(getData(dataFolder,dataset)) 
    df.loc[:,'exp_split'] = 'train'
    filename = 'diversityLSTM/Transparency/preprocess/ourData/'+dataset + '_dataset.csv'
    df.to_csv(filename, index=False)
    print('{} loaded'.format(dataset))