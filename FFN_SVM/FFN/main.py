import logging
import argparse
import os
from typing import *
import random

import numpy as np
import pandas as pd
import torch
import spacy

from utils import generateFastTextData_Spacy, Dataset, trainFFN
from preprocessing.preprocess import getData, splitData
from generation.feature import feature
from generation.partition import partition
from generation.deep import deep


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataFolder", required=True, type=str, help="The input data dir.")
    parser.add_argument(
        "--output_dir", required=True, type=str, 
        help="The output directory where the model checkpoints will be written.")
    parser.add_argument(
        "--task", required=True, type=str, 
        help="Choose whether to train the model or to generate lexicon.")
    parser.add_argument(
        "--method", type=str, help="Choose the method for lexicon generation."
    )
    parser.add_argument("--model_dir", type=str, help="The dir to the FFN model.")
    parser.add_argument("--masker", type=str, help="The masker used by partition shap.")
    parser.add_argument(
        "--background_size", default=100, type=int, help="Background size for Partition Masker and DeepShap.")

    args = parser.parse_args()

    return args


def train(dataset:str, nlp, args, device)-> None:
    logger.info("Start Training {}".format(dataset))

    trainDf, devDf, _ = splitData(getData(args.dataFolder, dataset))
    trainData = generateFastTextData_Spacy(trainDf, nlp, textVariable = "text")
    devData = generateFastTextData_Spacy(devDf, nlp, textVariable = "text")

    trainDataset = Dataset(trainDf, trainData)
    devDataset = Dataset(devDf, devData)

    NNnet = trainFFN(trainDataset, devDataset, max_epochs = 100, device=device)

    #Save the model
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output = os.path.join(args.output_dir,dataset+".bin")
        
    logger.info("Saving model to %s" % output)
    torch.save(NNnet.state_dict(), output)
        

def main()-> None:
    args = parse()

    nlp = spacy.load("./fasttext")

    ## Dataset that will be used for creating the lexicon
    train_data1 = ["nrc_joy", "yelp_subset","amazon_finefood_subset","amazon_toys_subset","empathy"]
    train_data2 = ["surprise", "sadness", "fear", "anger"]

    test_data1 = train_data1+["song_joy", "dialog_joy", "friends_joy", "emobank"]
    test_data2 = ["nrc","song","dialog","friends"]

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        logger.info("There are {} GPU(s) available.".format(torch.cuda.device_count()))
        logger.info("We will use the GPU: {}".format(torch.cuda.get_device_name(0)))

    else:
        logger.info("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    if args.task == "train":
        for data in train_data1:
            train(data, nlp, args, device)
        for data in train_data2:
            train("nrc_"+data, nlp, args, device)
    elif args.task == "generate":
        assert args.method is not None
        assert args.model_dir is not None
        
        results = []
        if args.method == "feature":
            for data in train_data1:
                results += feature(data, test_data1, nlp, args, device)
            for data in train_data2:
                results += feature(
                    "nrc_"+data, [i+"_"+data for i in test_data2], nlp, args, device)
        elif args.method == "deep":
            for data in train_data1:
                results += deep(data, test_data1, nlp, args, device)
            for data in train_data2:
                results += deep(
                    "nrc_"+data, [i+"_"+data for i in test_data2], nlp, args, device)
        elif args.method == "partition":
            assert args.masker is not None
            for data in train_data1:
                results += partition(data, test_data1, nlp, args, device)
            for data in train_data2:
                results += partition(
                    "nrc_"+data, [i+"_"+data for i in test_data2], nlp, args, device)
        else:
            logger.error("{} is not a valid method!".format(args.method))
        
        results = pd.DataFrame(results)
        results.columns = ["TrainData","TestData","modelAcc", "modelF1", "lexiconAcc", "lexiconF1"]
        results.to_csv("Results_{}.csv".format(args.method),index = False, index_label = False)
    else:
        logger.error("Wrong task!!!")


if __name__ == "__main__":
    main()
























