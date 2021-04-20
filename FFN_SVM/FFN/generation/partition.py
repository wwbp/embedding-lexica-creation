import logging
from typing import *

import torch

from utils import NNnet, generateLexicon_FFN, getLexicon, testFFN
from preprocessing.preprocess import getData, splitData


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def feature(train:str, test:List[str], nlp, args, device, background=None)->List[List[Union[str, float]]]:
    logger.info("Generating lexicon for {}".format(train))
    result = []

    NNnet = NNnet().load_state_dict(torch.load(args.model))
    trainDf, devDf, _ = splitData(getData(args.dataFolder, train))
    
    lexicon = generateLexicon_FFN(NNnet,trainDf,nlp,args.method,args.masker,background,device=device)
    outfilename = f"{args.output_dir}/{train}_ffn_feature.csv"
    lexicon.to_csv(outfilename, index = False, index_label = False)
    lexiconWords, lexiconMap = getLexicon(df = lexicon)
    
    logger.info("Running evaluation.")
    for dataset in test:
        result.append([train] + testFFN(
            NNnet, dataset, lexiconWords, lexiconMap, nlp, args.dataFolder, device))

    logger.info("Done!")
    return result

    