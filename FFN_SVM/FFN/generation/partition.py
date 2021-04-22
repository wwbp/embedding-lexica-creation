import logging
from typing import *
import pandas as pd

import torch

from utils import NNnet, generateLexicon_FFN, getLexicon, testFFN, generateFastTextData_Spacy
from preprocessing.preprocess import getData, splitData


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def partition(
    train:str, test:List[str], nlp, args, device)->List[List[Union[str, float]]]:
    logger.info("Generating lexicon for {}".format(train))
    if args.masker == "Partition":
        df_background = pd.read_csv("./FFN_SVM/FFN/backgrounds_500/"+train+"_500_bg.csv")
        assert args.background_size <= 500
        background = df_background.iloc[250-int(args.background_size/2):250+int(args.background_size/2)]
        background = generateFastTextData_Spacy(background, nlp)
    elif args.masker == "Text":
        background = None
    else:
        logger.error("{} is not a supported masker!".format(args.masker))
    
    result = []

    NNnet = NNnet().load_state_dict(torch.load(args.model+train+".bin"))
    trainDf, devDf, _ = splitData(getData(args.dataFolder, train))
    
    lexicon = generateLexicon_FFN(NNnet,trainDf,nlp,args.method,args.masker,background,device=device)
    outfilename = f"{args.output_dir}/{train}_ffn_ps.csv"
    lexicon.to_csv(outfilename, index = False, index_label = False)
    lexiconWords, lexiconMap = getLexicon(df = lexicon)
    
    logger.info("Running evaluation.")
    for dataset in test:
        result.append([train] + testFFN(
            NNnet, dataset, lexiconWords, lexiconMap, nlp, args.dataFolder, device))

    logger.info("Done!")
    return result

    