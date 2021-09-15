import pandas as pd 
import argparse
import os

parser = argparse.ArgumentParser(description='Run Preprocessing on dataset')
parser.add_argument('--data_folder', type=str, required=True)
args = parser.parse_args()

dataFolder = args.data_folder
train_dataset_list = ['nrc_joy','nrc_anger','nrc_sadness','nrc_surprise','nrc_fear','amazon_toys_subset','amazon_finefood_subset','yelp_subset']
for dataset in train_dataset_list:
    path = os.path.join(dataFolder, dataset)
    train = pd.read_csv(os.path.join(path, 'train.csv'))
    dev = pd.read_csv(os.path.join(path, 'dev.csv'))
    test = pd.read_csv(os.path.join(path, 'test.csv'))
    train.loc[:,'exp_split'] = 'train'
    test.loc[:,'exp_split'] = 'test'
    dev.loc[:,'exp_split'] = 'dev'
    df = pd.concat([train,test,dev])
    filename = 'diversityLSTM/Transparency/preprocess/ourData/'+dataset + '_dataset.csv'
    df.to_csv(filename, index=False)
    print('{} loaded'.format(dataset))
external_dataset_list = ['friends_joy','friends_anger','friends_sadness','friends_surprise','friends_fear','dialog_joy','dialog_anger','dialog_sadness','dialog_surprise','dialog_fear','song_joy','song_anger','song_sadness','song_surprise','song_fear','emobank']
for dataset in external_dataset_list:
    path = os.path.join(dataFolder, 'test_datasets')
    df = pd.read_csv(os.path.join(path, dataset+'.csv'))
    df.loc[:,'exp_split'] = 'train'
    filename = 'diversityLSTM/Transparency/preprocess/ourData/'+dataset + '_dataset.csv'
    df.to_csv(filename, index=False)
    print('{} loaded'.format(dataset))