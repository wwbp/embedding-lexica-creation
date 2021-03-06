import logging
import argparse
import random

import numpy as np
import pandas as pd
import torch
import spacy

from utils import *
from preprocessing.preprocess import *


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def train_generate(lexiconDataset, nlp):
    logger.info("Start Training {}".format(lexiconDataset))
    trainDf, devDf, _ = splitData(getData(args.dataFolder, lexiconDataset))
    trainData = generateFastTextData_Spacy(trainDf, nlp, textVariable = "text")
    devData = generateFastTextData_Spacy(devDf, nlp, textVariable = "text")

    trainDataset = Dataset(trainDf, trainData)
    devDataset = Dataset(devDf, devData)

    NNnet = trainFFN(trainDataset, devDataset, max_epochs = 100, device=device)

    lexicon = generateLexicon_FFN(NNnet,trainDf, nlp, device=device)

    lexicon.rename({"NNprob":"score"},axis = 1, inplace = True)

    return NNnet, lexicon


if __name__ == "__main__":
    ######### USER INPUTS ###########

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataFolder", required=True, type=str, help="The input data dir.")
    parser.add_argument("--output_dir", required=True, type=str, help="The output directory where the model checkpoints will be written.")

    args = parser.parse_args()

    nlp = spacy.load("./fasttext")

    ## Dataset that will be used for creating the lexicon
    lexiconDataset1 = ["nrc_joy", "yelp_subset","amazon_finefood_subset","amazon_toys_subset","empathy"]
    lexiconDataset2 = ["surprise", "sadness", "fear", "anger"]

    dataList1 = ["song_joy", "dialog_joy", "friends_joy", "emobank"]
    dataList2 = ["nrc","song","dialog","friends"]

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        logger.info("There are {} GPU(s) available.".format(torch.cuda.device_count()))
        logger.info("We will use the GPU: {}".format(torch.cuda.get_device_name(0)))

    else:
        logger.info("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    ##################################

    results = []
    for lexica in lexiconDataset1:
        NNnet, lexicon = train_generate(lexica, nlp)
        outfilename = f"{args.output_dir}/{lexica}_ffn_feature.csv"
        lexicon.to_csv(outfilename, index = False, index_label = False)

        lexiconWords, lexiconMap = getLexicon(df = lexicon)

        for data in dataList1+lexiconDataset1:
            results.append([lexica]+testFFN(NNnet,data,lexiconWords, lexiconMap, nlp, args.dataFolder, device))

    for lexica in lexiconDataset2:
        NNnet, lexicon = train_generate("nrc_"+lexica, nlp)
        outfilename = f"{args.output_dir}/nrc_{lexica}_ffn_feature.csv"
        lexicon.to_csv(outfilename, index = False, index_label = False)

        lexiconWords, lexiconMap = getLexicon(df = lexicon)

        for data in dataList2:
            results.append([lexica]+testFFN(
                NNnet,data+"_"+lexica,lexiconWords, lexiconMap, nlp, args.dataFolder, device))
            
    results = pd.DataFrame(results)
    results.columns = ["TrainData","TestData","modelAcc", "modelF1", "lexiconAcc", "lexiconF1"]
    results.to_csv("Results_FFN.csv",index = False, index_label = False)

    logger.info("Done")
























