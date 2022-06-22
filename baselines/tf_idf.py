import logging
import argparse
import os
from typing import *
import random

import numpy as np
import pandas as pd
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer

from utils import getLexicon, testUni

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


random.seed(42)
np.random.seed(42)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataFolder", required=True, type=str, help="The input data dir.")
    parser.add_argument("--output_dir", type=str, help="The dir to the FFN model.")

    args = parser.parse_args()

    return args


def generate(train:str, test:List[str], nlp, fast, args):
    logger.info("Generating lexicon for {}".format(train))
    result = []

    path = os.path.join(args.dataFolder, train)
    trainDf = pd.read_csv(os.path.join(path, 'train.csv'))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    outfilename = f"{args.output_dir}/{train}_idf.csv"
    
    if os.path.exists(outfilename):
        logger.info("File already exists, skipped!")
        lexicon = pd.read_csv(outfilename)
    else:
        scores = nlp.fit_transform(trainDf[trainDf.label==1].text)
        scores = scores.sum(axis=0) / scores.getnnz(axis=0)
        lexicon1 = pd.DataFrame(np.array(scores)[0], 
                  index=nlp.get_feature_names(), 
                  columns=["Value"])
        lexicon1.sort_values('Value', ascending=False, inplace=True)
        lexicon1.reset_index(inplace=True)
        lexicon1.rename(columns = {'index':'Word'}, inplace = True)
        scores = nlp.fit_transform(trainDf[trainDf.label==0].text)
        scores = scores.sum(axis=0) / scores.getnnz(axis=0)
        lexicon = pd.DataFrame(np.array(scores)[0], 
                  index=nlp.get_feature_names(), 
                  columns=["Value"])
        lexicon.sort_values('Value', ascending=False, inplace=True)
        lexicon.reset_index(inplace=True)
        lexicon.rename(columns = {'index':'Word'}, inplace = True)
        lexicon = lexicon.merge(lexicon1, how='outer', on='Word')
        lexicon = lexicon.fillna(0)
        lexicon['Value'] = lexicon.Value_x - lexicon.Value_y
        lexicon = lexicon[['Word', 'Value']]
        lexicon.to_csv(outfilename)

    lexiconWords, lexiconMap = getLexicon(df = lexicon)

    logger.info("Running evaluation.")
    for dataset in test:
        result.append([train] + testUni(
            dataset, lexiconWords, lexiconMap, fast, args.dataFolder))

    logger.info("Done!")
    return result


def main():
    args = parse()

    nlp = TfidfVectorizer(use_idf=True)
    fast = spacy.load('./fasttext')

    ## Dataset that will be used for creating the lexicon
    train_data1 = ["nrc_joy", "yelp_subset","amazon_finefood_subset","amazon_toys_subset",]
    train_data2 = ["surprise", "sadness", "fear", "anger"]

    test_data1 = train_data1+["song_joy", "dialog_joy", "friends_joy", "emobank"]
    test_data2 = ["nrc","song","dialog","friends"]

    results = []
    for data in train_data1:
        results += generate(data, test_data1, nlp, fast, args)
    for data in train_data2:
        results += generate(
                "nrc_"+data, [i+"_"+data for i in test_data2], nlp, fast, args)
         
    results = pd.DataFrame(results)
    results.columns = ["TrainData","TestData","lexiconAcc", "lexiconF1"]
    results.to_csv("Results_idf.csv",index = False, index_label = False)


if __name__ == "__main__":
    main()