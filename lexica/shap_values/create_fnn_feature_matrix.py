# -*- coding: utf-8 -*-
import sys, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk import word_tokenize
import nltk
nltk.download('punkt')

# This is for downloading the word vectors
import wget

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Concatenate, Conv1D, MaxPool1D, Reshape, Flatten, AveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import shap

VISUALIZE_MODEL=True

if not os.path.exists("empathic_reactions"):
    os.system("git clone https://github.com/wwbp/empathic_reactions")
sys.path.insert(1, "empathic_reactions")

# NOTE: variables ending in df are pandas dataframes
empathic_reactions_df = pd.read_csv('empathic_reactions/data/responses/data/messages.csv')

"""###Next we'll get the word embeddings"""


if not os.path.exists("crawl-300d-2M.vec.zip"):
   wget.download('https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip')

from modeling.embedding import Embedding

embs = Embedding.from_fasttext_vec(
    path='./crawl-300d-2M.vec.zip',
    zipped=True,
    file='crawl-300d-2M.vec', 
    vocab_limit=100_000)

"""##Now we start to setup the model finally"""

import modeling.feature_extraction as fe
TIMESTEPS=200

FEATURES_MATRIX=fe.embedding_centroid(empathic_reactions_df['essay'], embs)
print(FEATURES_MATRIX.shape)
np.savetxt('fnn_feature_matrix.txt', FEATURES_MATRIX)

#import pdb;pdb.set_trace(
