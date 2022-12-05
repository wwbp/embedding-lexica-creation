import logging
import argparse
import os
from typing import *
import random

import numpy as np
import pandas as pd
import torch
import spacy

from utils import generateFastTextData_Spacy, Dataset, trainFFN, NNNet, generateLexicon_FFN, getLexicon, testFFN


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
        "--model_dir", required=True, type=str, 
        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--lexicon_dir", type=str, help="The dir to the FFN model.")
    parser.add_argument("--train_model", action="store_true", help="Train new FFN models or not.")

    args = parser.parse_args()

    return args


def train(dataset:str, nlp, args, device)-> None:
    model_dir = args.model_dir
    output = os.path.join(model_dir,dataset+".bin")

    if args.train_model:
        logger.info("Start Training {}".format(dataset))

        path = os.path.join(args.dataFolder, dataset)
        trainDf = pd.read_csv(os.path.join(path, 'train.csv'))
        devDf = pd.read_csv(os.path.join(path, 'dev.csv'))

        trainData = generateFastTextData_Spacy(trainDf, nlp, textVariable = "text")
        devData = generateFastTextData_Spacy(devDf, nlp, textVariable = "text")

        trainDataset = Dataset(trainDf, trainData)
        devDataset = Dataset(devDf, devData)

        NNnet = trainFFN(trainDataset, devDataset, max_epochs = 100, device=device)

        #Save the model
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        logger.info("Saving model to %s" % output)
        torch.save(NNnet.state_dict(), output)
    else:
        NNnet = NNNet()
        NNnet.load_state_dict(torch.load(output))

    return NNnet
        

def generate(train:str, test:List[str], nlp, args, device, model=None)->List[List[Union[str, float]]]:
    logger.info("Generating lexicon for {}".format(train))
    result = []

    if model is None:
        NNnet = NNNet()
        NNnet.load_state_dict(torch.load(args.model_dir+"/"+train+".bin"))
    else:
        NNnet = model
    path = os.path.join(args.dataFolder, train)
    trainDf = pd.read_csv(os.path.join(path, 'train.csv'))

    if not os.path.exists(args.lexicon_dir):
        os.makedirs(args.lexicon_dir)

    outfilename = f"{args.lexicon_dir}/{train}_ffn_feature.csv"
    
    if os.path.exists(outfilename):
        logger.info("File already exists, skipped!")
        lexicon = pd.read_csv(outfilename)
    else:
        lexicon = generateLexicon_FFN(NNnet,trainDf,nlp,device=device)
        lexicon.to_csv(outfilename)
    lexiconWords, lexiconMap = getLexicon(df = lexicon)
    
    logger.info("Running evaluation.")
    for dataset in test:
        result.append([train] + testFFN(
            NNnet, dataset, lexiconWords, lexiconMap, nlp, args.dataFolder, device))

    logger.info("Done!")
    return result


def main()-> None:
    args = parse()

    nlp = spacy.load("../fasttext")

    ## Dataset that will be used for creating the lexicon
    train_data1 = ["nrc_joy", "yelp_subset","amazon_finefood_subset","amazon_toys_subset",]
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

    results = []
    for data in train_data1:
        model = train(data, nlp, args, device)
        results += generate(data, test_data1, nlp, args, device, model)
    for data in train_data2:
        model = train("nrc_"+data, nlp, args, device)
        results += generate(
                "nrc_"+data, [i+"_"+data for i in test_data2], nlp, args, device, model)
         
    results = pd.DataFrame(results)
    results.columns = ["TrainData","TestData","modelAcc", "modelF1", "lexiconAcc", "lexiconF1"]
    results.to_csv("Results_FFN.csv",index = False, index_label = False)


if __name__ == "__main__":
    main()
























