import os
import sys
import logging

sys.path.append(os.getcwd())

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


from bs4 import BeautifulSoup


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# from html.parser import HTMLParser
# class HTMLStripper(HTMLParser):
#     def __init__(self):
#         super().__init__()
#         self.reset()
#         self.fed = []
#     def handle_data(self, d):
#         self.fed.append(d)
#     def get_data(self):
#         return ' '.join([x for x in self.fed if not x.isspace()])


subset_size = 300000


def getData(dataFolder, dataset):
    '''
    Available options for dataset
        -----------------------
        empathy
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
        emobank
        -----------------------
        sst2
    '''
#-----------------------------------------------------------------------------------------------
    
    if dataset == 'empathy':
        
        df = pd.read_csv(os.path.join(dataFolder,'Empathy/messages.csv'))
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.empathy_bin
        df['text'] = df.essay
        df = df.filter(['text','label'], axis = 1)
        
#-----------------------------------------------------------------------------------------------
    
    elif dataset == 'yelp':

        df = pd.read_csv(os.path.join(dataFolder,'Yelp/df1M.tsv'), sep='\t')
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df = df[df.stars!=3]
        df['label'] = df.stars.apply(lambda x : 0 if x<3 else 1)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'yelp_subset':
        
        df = pd.read_csv(os.path.join(dataFolder,'Yelp/df1M.tsv'), sep='\t')
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df = df[df.stars!=3]
        df['label'] = df.stars.apply(lambda x : 0 if x<3 else 1)
        df = df.filter(['text','label'], axis = 1)
        df = df.iloc[0:subset_size]
    
#-----------------------------------------------------------------------------------------------
    
    elif dataset == 'amazon_finefood':
        
        df = pd.read_csv(os.path.join(dataFolder, 'AmazonFineFood/Reviews.csv'))
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df = df[df.Score!=3]
        df['label'] = df.Score.apply(lambda x : 0 if x<3 else 1)
        df['text'] = df.Text
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'amazon_finefood_subset':
        
        df = pd.read_csv(os.path.join(dataFolder, 'AmazonFineFood/Reviews.csv'))
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df = df[df.Score!=3]
        df['label'] = df.Score.apply(lambda x : 0 if x<3 else 1)
        df['text'] = df.Text
        df = df.filter(['text','label'], axis = 1)  
        df = df.iloc[0:subset_size]  

    elif dataset == 'amazon_toys':
        
        df = pd.read_json(os.path.join(dataFolder, 'AmazonProductData/reviews_Toys_and_Games_5.json'), lines = True)
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df = df[df.overall!=3]
        df['label'] = df.overall.apply(lambda x : 0 if x<3 else 1)
        df['text'] = df.reviewText
        df = df.filter(['text','label'], axis = 1)  

        
    elif dataset == 'amazon_toys_subset':
        
        df = pd.read_json(os.path.join(dataFolder, 'AmazonProductData/reviews_Toys_and_Games_5.json'), lines = True)
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df = df[df.overall!=3]
        df['label'] = df.overall.apply(lambda x : 0 if x<3 else 1)
        df['text'] = df.reviewText
        df = df.filter(['text','label'], axis = 1)  
        # df = df.iloc[0:subset_size]
        
#-----------------------------------------------------------------------------------------------
        
    elif dataset == 'nrc_joy':
        
        df = pd.read_csv(os.path.join(dataFolder,'NRCData/msgs_tec_langClean.csv'))
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.emotion.apply(lambda x : 1 if x == 'joy' else 0)
        df['text'] = df.message
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'nrc_sadness':
        
        df = pd.read_csv(os.path.join(dataFolder,'NRCData/msgs_tec_langClean.csv'))
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.emotion.apply(lambda x : 1 if x == 'sadness' else 0)
        df['text'] = df.message
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'nrc_fear':
        
        df = pd.read_csv(os.path.join(dataFolder,'NRCData/msgs_tec_langClean.csv'))
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.emotion.apply(lambda x : 1 if x == 'fear' else 0)
        df['text'] = df.message
        df = df.filter(['text','label'], axis = 1)

        
    elif dataset == 'nrc_anger':
        
        df = pd.read_csv(os.path.join(dataFolder,'NRCData/msgs_tec_langClean.csv'))
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.emotion.apply(lambda x : 1 if x == 'anger' else 0)
        df['text'] = df.message
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'nrc_surprise':
        
        df = pd.read_csv(os.path.join(dataFolder,'NRCData/msgs_tec_langClean.csv'))
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.emotion.apply(lambda x : 1 if x == 'surprise' else 0)
        df['text'] = df.message
        df = df.filter(['text','label'], axis = 1)
        
#-----------------------------------------------------------------------------------------------

    elif dataset == 'song_joy':
        
        df = pd.read_csv(os.path.join(dataFolder,'SongLyrics/RadaEmotionWheelVersesCleaned.csv'))
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.joy.apply(lambda x : 1 if x == 1.0 else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'song_sadness':
        
        df = pd.read_csv(os.path.join(dataFolder,'SongLyrics/RadaEmotionWheelVersesCleaned.csv'))
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.sadness.apply(lambda x : 1 if x == 1.0 else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'song_fear':
        
        df = pd.read_csv(os.path.join(dataFolder,'SongLyrics/RadaEmotionWheelVersesCleaned.csv'))
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.fear.apply(lambda x : 1 if x == 1.0 else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'song_anger':
        
        df = pd.read_csv(os.path.join(dataFolder,'SongLyrics/RadaEmotionWheelVersesCleaned.csv'))
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.anger.apply(lambda x : 1 if x == 1.0 else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'song_surprise':
        
        df = pd.read_csv(os.path.join(dataFolder,'SongLyrics/RadaEmotionWheelVersesCleaned.csv'))
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.surprise.apply(lambda x : 1 if x == 1.0 else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'song_disgust':
        
        df = pd.read_csv(os.path.join(dataFolder,'SongLyrics/RadaEmotionWheelVersesCleaned.csv'))
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.disgust.apply(lambda x : 1 if x == 1.0 else 0)
        df = df.filter(['text','label'], axis = 1)
        
#-----------------------------------------------------------------------------------------------
        
    elif dataset == 'dialog_joy':
        
        df = pd.read_csv(os.path.join(dataFolder,'DailyDialog/dialogData.csv'), sep='\t')
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.label.apply(lambda x : 1 if x == 'joy' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'dialog_sadness':
        
        df = pd.read_csv(os.path.join(dataFolder,'DailyDialog/dialogData.csv'), sep='\t')
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.label.apply(lambda x : 1 if x == 'sadness' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'dialog_fear':
        
        df = pd.read_csv(os.path.join(dataFolder,'DailyDialog/dialogData.csv'), sep='\t')
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.label.apply(lambda x : 1 if x == 'fear' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'dialog_anger':
        
        df = pd.read_csv(os.path.join(dataFolder,'DailyDialog/dialogData.csv'), sep='\t')
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.label.apply(lambda x : 1 if x == 'anger' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'dialog_surprise':
        
        df = pd.read_csv(os.path.join(dataFolder,'DailyDialog/dialogData.csv'), sep='\t')
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.label.apply(lambda x : 1 if x == 'surprise' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'dialog_disgust':
        
        df = pd.read_csv(os.path.join(dataFolder,'DailyDialog/dialogData.csv'), sep='\t')
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.label.apply(lambda x : 1 if x == 'disgust' else 0)
        df = df.filter(['text','label'], axis = 1)
        
#-----------------------------------------------------------------------------------------------

    elif dataset == 'friends_joy':
        
        df = pd.read_csv(os.path.join(dataFolder,'EmotionLines/friendsData.csv'), sep='\t')
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.label.apply(lambda x : 1 if x == 'joy' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'friends_sadness':
        
        df = pd.read_csv(os.path.join(dataFolder,'EmotionLines/friendsData.csv'), sep='\t')
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.label.apply(lambda x : 1 if x == 'sadness' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'friends_fear':
        
        df = pd.read_csv(os.path.join(dataFolder,'EmotionLines/friendsData.csv'), sep='\t')
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.label.apply(lambda x : 1 if x == 'fear' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'friends_anger':
        
        df = pd.read_csv(os.path.join(dataFolder,'EmotionLines/friendsData.csv'), sep='\t')
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.label.apply(lambda x : 1 if x == 'anger' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'friends_surprise':
        
        df = pd.read_csv(os.path.join(dataFolder,'EmotionLines/friendsData.csv'), sep='\t')
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.label.apply(lambda x : 1 if x == 'surprise' else 0)
        df = df.filter(['text','label'], axis = 1)
        
    elif dataset == 'friends_disgust':
        
        df = pd.read_csv(os.path.join(dataFolder,'EmotionLines/friendsData.csv'), sep='\t')
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df['label'] = df.label.apply(lambda x : 1 if x == 'disgust' else 0)
        df = df.filter(['text','label'], axis = 1)
        
#-----------------------------------------------------------------------------------------------
        
    elif dataset == 'emobank':
        
        df = pd.read_csv(os.path.join(dataFolder,'Empathy/emobank.csv'))
        df.replace('', float('NaN'), inplace=True)
        df.dropna(inplace=True)
        df = df[df.V!=3]
        df['label'] = df.V.apply(lambda x : 0 if x<3 else 1)
        df['text'] = df.essay
        df = df.filter(['text','label'], axis = 1)

#-----------------------------------------------------------------------------------------------

    elif dataset == 'sst2':

        df = []
        
        def process(data):
            data = pd.read_csv(os.path.join(dataFolder, 'SST-2/'+data+'.tsv'), sep='\t')
            data.replace('', float('NaN'), inplace=True)
            data.dropna(inplace=True)
            data.columns = ['text', 'label']
            return data
        
        df.append(process('train'))
        df.append(process('dev'))
        df.append(process('test'))

    return df


def splitData(df, balanceTrain = True):
    trainX, tempX, _, _ = train_test_split(df, df.label, test_size = 0.2, random_state = 42)
    devX, testX, _, _ = train_test_split(tempX, tempX.label, test_size = 0.5, random_state = 42)
    if balanceTrain:
        trainX = balanceData(trainX)
        testX = balanceData(testX)
        devX = balanceData(devX)
    return trainX, devX, testX  


def balanceData(df):
    ros = RandomUnderSampler(random_state = 42)
    df, _ = ros.fit_resample(df, df.label)
    return df


def strip_html(html):
#     s = HTMLStripper()
#     s.feed(html)
#     return s.get_data()
    soup = BeautifulSoup(html,features='html.parser')
    return soup.get_text()


def processData(df):
    df.text = df.text.apply(strip_html)
#     df.replace('', float('NaN'), inplace=True)
#     df.dropna(inplace=True)
    
    return df


if __name__ == '__main__':
    
    dataFolder = 'Raw'
    output = 'cleandata'
    emotions = ['joy', 'sadness', 'surprise', 'anger', 'fear']
    train_datasets = [i + '_subset' for i in ['yelp', 'amazon_finefood', 'amazon_toys']]+\
        ['nrc_'+i for i in emotions]
    test_datasets = ['song_'+i for i in emotions]+['friends_'+i for i in emotions]+\
        ['dialog_'+i for i in emotions]+['emobank']
    
    # for dataset in train_datasets:
    for dataset in ['amazon_toys_subset']:
        logger.info('processing {}'.format(dataset))
        if dataset == 'sst2':
            cleandata = getData(dataFolder, dataset)
        else:
            data = getData(dataFolder, dataset)
            cleandata = []
            data = processData(data)
            data = data.dropna()
            for split in splitData(data):
                cleandata.append(split)

        path = os.path.join(output, dataset)
        if not os.path.exists(path):
            os.mkdir(path)

        cleandata[0].to_csv(os.path.join(path, 'train.csv'), index=None)
        cleandata[1].to_csv(os.path.join(path, 'dev.csv'), index=None)
        cleandata[2].to_csv(os.path.join(path, 'test.csv'), index=None)
    
    # for dataset in test_datasets:
    #     logger.info('processing {}'.format(dataset))
    #     data = getData(dataFolder, dataset)
    #     data = processData(data)
    #     data = data.dropna()
    #     data = balanceData(data)

    #     path = os.path.join(output, 'test_datasets')
    #     if not os.path.exists(path):
    #         os.mkdir(path)

    #     data.to_csv(os.path.join(path, dataset+'.csv'), index=None)

