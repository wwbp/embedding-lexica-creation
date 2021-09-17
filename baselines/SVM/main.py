import argparse
import logging
import random
import os

import numpy as np
import pandas as pd
import spacy

from utils import *


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


random.seed(42)
np.random.seed(42)


def train_generate(lexiconDataset, nlp):
    logger.info("Start Training {}".format(lexiconDataset))
    path = os.path.join(args.dataFolder, lexiconDataset)
    trainDf = pd.read_csv(os.path.join(path, 'train.csv'))
    trainData = generateFastTextData_Spacy(trainDf, nlp, textVariable = "text")

    model = trainSVM(trainData, trainDf)

    lexiconDf = generateLexicon_SVM(model, trainDf, nlp)

    return model, lexiconDf


if __name__ == "__main__":
    ######### USER INPUTS ###########

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataFolder", required=True, type=str, help="The input data dir.")
    parser.add_argument("--output_dir", required=True, type=str, help="The output directory where the model checkpoints will be written.")

    args = parser.parse_args()

    nlp = spacy.load("./fasttext")

    ## Dataset that will be used for creating the lexicon
    lexiconDataset1 = ["nrc_joy", "yelp_subset","amazon_finefood_subset","amazon_toys_subset"]
    lexiconDataset2 = ["surprise", "sadness", "fear", "anger"]

    dataList1 = ["song_joy", "dialog_joy", "friends_joy", "emobank"]
    dataList2 = ["nrc","song","dialog","friends"]
    ##################################

    results = []
    for lexica in lexiconDataset1:
        model, lexiconDf = train_generate(lexica, nlp)
        outfilename = f"{args.output_dir}/{lexica}_svm_feature.csv"
        lexiconDf.to_csv(outfilename)

        lexiconWords, lexiconMap = getLexicon(df = lexiconDf)
        
        for data in dataList1+lexiconDataset1:
            results.append([lexica]+ testSVM(model, data,lexiconWords, lexiconMap, nlp, args.dataFolder))
        
    for lexica in lexiconDataset2:
        model, lexiconDf = train_generate("nrc_"+lexica, nlp)
        outfilename = f"{args.output_dir}/nrc_{lexica}_svm_feature.csv"
        lexiconDf.to_csv(outfilename)

        lexiconWords, lexiconMap = getLexicon(df = lexiconDf)

        for data in dataList2:
            results.append([lexica]+ testSVM(
                model, data+"_"+lexica,lexiconWords, lexiconMap, nlp, args.dataFolder))

    results = pd.DataFrame(results)
    results.columns = ["TrainData","TestData","modelAcc", "modelF1", "lexiconAcc", "lexiconF1"]
    results.to_csv("Results_SVM.csv",index = False, index_label = False)

    logger.info("Done")
