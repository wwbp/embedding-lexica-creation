import os
import sys
import logging

sys.path.append(os.getcwd())

import pandas as pd


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    
    output = "cleandata"
    emotions = ["joy", "sadness", "surprise", "anger", "fear"]
    train_datasets = [i + "_subset" for i in ["yelp", "amazon_finefood", "amazon_toys"]]+\
        ["nrc_"+i for i in emotions]
    
    outputpath = ("background")
    if not os.path.exists(outputpath):
        os.mkdir(outputpath)

    for dataset in train_datasets:
        logger.info("processing {}".format(dataset))

        path = os.path.join(output, dataset)
        path = os.path.join(path, "train.csv")

        data = pd.read_csv(path)
        pos_sample = data[data.label == 0].sample(250, random_state=42, axis=0)
        neg_sample = data[data.label == 1].sample(250, random_state=42, axis=0)
        sample = pd.concat([pos_sample, neg_sample], axis=0).sample(frac=1)
        
        sample.to_csv(os.path.join(outputpath, dataset)+".csv", index=None)