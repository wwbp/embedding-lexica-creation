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

VISUALIZE_MODEL=False

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
    file='crawl-300d-2M.vec')

"""##Now we start to setup the model finally"""

import modeling.feature_extraction as fe
TIMESTEPS=200

FEATURES_MATRIX=fe.embedding_matrix(empathic_reactions_df['essay'], embs, TIMESTEPS)
print(FEATURES_MATRIX.shape)

# NOTE this is from modeling.feature_extraction.get_cnn
# https://github.com/wwbp/empathic_reactions/blob/master/modeling/common.py#L126
# but we need tf.keras instead of keras for the new SHAP
# so I've copied and pasted here.

def get_cnn(input_shape, num_outputs, num_filters, learning_rate, dropout_conv, problem):
	
	if problem not in ['regression', 'classification']:
		raise ValueError
	
	# loosely based on https://github.com/bhaveshoswal/CNN-text-classification-keras/blob/master/model.py

	# filter_sizes=[3,4,5]
	embedding_dim=input_shape[1]
	sequence_length=input_shape[0]

	

	l2_strength=.001

	inputs = Input(shape=input_shape)
	inputs_drop = Dropout(dropout_conv)(inputs)

	filter_size=1
	conv_1=Conv1D(filters=num_filters, kernel_size=filter_size, strides=1, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(inputs_drop)
	pool_1=AveragePooling1D(pool_size=input_shape[0]-filter_size+1, strides=1)(conv_1)
	pool_drop_1=Dropout(dropout_conv)(pool_1)

	filter_size=2
	conv_2=Conv1D(filters=num_filters, kernel_size=filter_size, strides=1, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(inputs_drop)
	pool_2=AveragePooling1D(pool_size=input_shape[0]-filter_size+1, strides=1)(conv_2)
	pool_drop_2=Dropout(dropout_conv)(pool_2)
	
	filter_size=3
	conv_3=Conv1D(filters=num_filters, kernel_size=filter_size, strides=1, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(inputs_drop)
	pool_3=AveragePooling1D(pool_size=input_shape[0]-filter_size+1, strides=1)(conv_3)
	pool_drop_3=Dropout(dropout_conv)(pool_3)

	concatenated=Concatenate(axis=1)([pool_drop_1, pool_drop_2, pool_drop_3])

	dense = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(l2_strength))(Flatten()(concatenated))
	dense_drop = Dropout(.5)(dense)
	
	if problem=='classification':
		output = Dense(units=num_outputs, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_strength))(dense_drop)
	if problem=='regression':
		output = Dense(units=num_outputs, activation=None, kernel_regularizer=regularizers.l2(l2_strength))(dense_drop)
	# this creates a model that includes
	model = Model(inputs=inputs, outputs=output)
	optimizer=Adam(lr=learning_rate)

	if problem=='regression':
		model.compile(loss='mse', optimizer=optimizer)
	if problem=='classification':
		model.compile(loss='binary_crossentropy', optimizer=optimizer)
	return model

model = get_cnn(
        input_shape=[TIMESTEPS,300], 
        num_outputs=1, 
        num_filters=100, 
        learning_rate=1e-3,
        dropout_conv=.5, 
        problem='regression')

print(model.summary())

if VISUALIZE_MODEL:
    # Plot model graph
    from tensorflow.keras.utils import plot_model
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')


"""##Now we've for the Empathic Reactions Dataset ready. Next step is to train a model to predict empathy (a.k.a. empathic concern) and distress (a.k.a. personal distress)"""

from time import time
start = time()
model.fit(FEATURES_MATRIX, 
		  empathic_reactions_df['empathy'],
		  epochs=50,
		  batch_size=32, 
		  validation_split=.1)
print(f'Seconds: {time()-start}')

"""##Now at long last we run SHAP DeepExplainer
**WARNING: This code is only meant as a starting point!**
"""

explainer = shap.DeepExplainer(model, FEATURES_MATRIX[:200])

word_idx_matrix = np.zeros((empathic_reactions_df['essay'].shape[0], TIMESTEPS), dtype=int)
tokenized_essays = empathic_reactions_df['essay'].apply(word_tokenize)
for i, sent in enumerate(tokenized_essays):
    sent=[str.lower() for str in sent]
    word_idx_matrix[i][TIMESTEPS-len(sent):TIMESTEPS] = [embs.wi.get(_, 0) for _ in sent]

from time import time
start = time()
shap_values = explainer.shap_values(FEATURES_MATRIX)
print(f'Seconds: {time()-start}')

word2values={}
shap_means = shap_values[0].mean(axis=2)
for i in range(FEATURES_MATRIX.shape[0]):
    for j in range(FEATURES_MATRIX.shape[1]):
        if word_idx_matrix[i][j] == 0:
            continue
        if embs.iw[word_idx_matrix[i][j]] not in word2values: 
            word2values[embs.iw[word_idx_matrix[i][j]]] = []
        word2values[embs.iw[word_idx_matrix[i][j]]].append(shap_means[i][j])

lexicon = {'words':[], 'values':[]}
min_word = ''
max_word = ''
min = 100
max = -100
for word in word2values:
    lexicon['words'].append(word)
    lexicon['values'].append(np.mean(word2values[word]))
    #print(word, np.mean(word2value[word]))
    if np.mean(word2values[word]) < min:
        min = np.mean(word2values[word])
        min_word = word
    if np.mean(word2values[word]) > max:
        max = np.mean(word2values[word])
        max_word=word

print(min_word, min)
print(max_word, max)

lexicon_df = pd.DataFrame.from_dict(lexicon)
print(lexicon_df.sort_values(by='values'))
