import pandas as pd
import numpy as np
import seaborn as sns
import os

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords 

from transformers import BertForSequenceClassification, BertTokenizer, BertForMaskedLM

from simpletransformers.language_modeling import LanguageModelingModel

from sklearn.metrics.pairwise import cosine_similarity, paired_euclidean_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import itertools
from collections import Counter
import sys
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
import sister


import spacy


from sklearn.svm import LinearSVC
from typing import List
from pathlib import Path
from joblib import dump, load

import plotly.graph_objects as go
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

######### USER INPUTS ###########

from utils import *
from preprocessing.preprocess import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataFolder", required=True, type=str, help="The input data dir.")
parser.add_argument("--output_dir", required=True, type=str, help="The output directory where the model checkpoints will be written.")

args = parser.parse_args()

nlp = spacy.load('./fasttext')

dataList = ['nrc_joy', 'yelp_subset','amazon_finefood_subset','amazon_toys_subset','empathy']

if torch.cuda.is_available():       
    device = torch.device("cuda")
    logger.info('There are {} GPU(s) available.'.format(torch.cuda.device_count()))
    logger.info('We will use the GPU: {}'.format(torch.cuda.get_device_name(0)))

else:
    logger.info('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

##################################


for data in dataList:
    trainDf, devDf, testDf = splitData(getData(args.dataFolder, data))
    trainData = generateFastTextData_Spacy(trainDf, nlp, textVariable = 'text')
    testData = generateFastTextData_Spacy(testDf, nlp, textVariable = 'text')

    model = trainSVM(trainData, testData, trainDf, testDf)

    lexiconDf = generateLexicon_SVM(model,trainDf, nlp)
    
    outfilename = f"{args.output_dir}/{data}_lexicon.csv"
    
    lexiconDf.to_csv(outfilename, index = False, index_label = False)


