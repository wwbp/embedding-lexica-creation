import argparse
import logging
import copy
import os

import numpy as np
import pandas as pd
from scipy import stats
from deep_shap import DeepExplainer
import spacy
import tokenizations

import torch
from transformers import BertTokenizerFast, DistilBertTokenizerFast

from utils.preprocess import getData, splitData
from utils.utils import get_dataset
from models.modeling_bert import BertForSequenceClassification
from models.modeling_distilbert import DistilBertForSequenceClassification


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()

parser.add_argument("--dataFolder", required=True, type=str, help="The input data dir.")
parser.add_argument("--dataset", required=True, type=str, help="The dataset we use.")
parser.add_argument("--output_dir", required=True, type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--task", required=True, type=str, help="Classification or regression.")

parser.add_argument("--model_kind", required=True, type=str)
parser.add_argument("--model", required=True, type=str, help="The pretrained Bert model we choose.")
parser.add_argument("--do_lower_case", action="store_true",
                        help= "Whether to lower case the input text. Should be True for uncased \
                            models and False for cased models.")
parser.add_argument("--tokenizer", type=str, help="Dir to tokenizer for prediction.")
parser.add_argument("--do_alignment", action="store_true")

parser.add_argument("--max_seq_length", type=int, default=128,
                    help="The maximum total input sequence length after WordPiece tokenization. ")

parser.add_argument("--gold_word", type=str, default=None, help="Gold word rating for evaluation.")

args = parser.parse_args()


def get_word_rating(model, input_ids, word_embeddings, attention_masks, tokenizer ,gold, device):
    
    logging.info('Getting lexicon')
    
    logging.info('Getting Shapley values')
    explainer = DeepExplainer(model, {'inputs_embeds':word_embeddings[:50].to(device)})
    shap_values = torch.from_numpy(explainer.shap_values({'inputs_embeds':word_embeddings[:5]})).sum(axis=-1)
    logging.info("Calculated done!")
    
    word2values = {}
    exclude = ['[CLS]', '[SEP]', '[PAD]']
    
    input_ids = torch.masked_select(input_ids[:5], attention_masks[:5].bool())
    words = tokenizer.convert_ids_to_tokens(input_ids)
    shap_values = torch.masked_select(shap_values, attention_masks[:5].bool())
    
    for index, word in enumerate(words):
        if word not in exclude:
            if word not in word2values:
                word2values[word] = [shap_values[index]]
            else:
                word2values[word].append(shap_values[index])
    
    lexicon = {'Word':[], 'Value': [], 'Value_sum': []}
    for word in word2values:
        lexicon['Word'].append(word)
        lexicon['Value'].append(np.mean(word2values[word]))
        lexicon['Value_sum'].append(np.sum(word2values[word]))
    
    lexicon_df = pd.DataFrame.from_dict(lexicon).sort_values(by='Value')
    logger.info(lexicon_df)
    if gold is not None:
        gold = pd.read_csv(gold, index_col=0)
        gold = gold[['Word', args.task+'.Mean.Sum']]
        
        merge_df = pd.merge(lexicon_df, gold, how='inner', on=['Word'])
        logger.info(merge_df)
        pearson, _ = stats.pearsonr(merge_df['Value'], merge_df[args.task+'.Mean.Sum'])
        logging.info("Pearson for word is %f" % pearson)
    
    logging.info('Writing to %s' % args.output)
    
    merge_df.to_csv(args.output)
    
    logging.info('Done!')
        

if __name__=="__main__":
    
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        logging.info('There are %d GPU(s) available.' % torch.cuda.device_count())
        logging.info('We will use the GPU: %s' % str(torch.cuda.get_device_name(0)))

    else:
        logging.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        
    df = getData(args.dataFolder, args.dataset, args.task)
    logger.info("Total Data:{}".format(df.shape[0]))
    if args.task == "classification":
        df_train, df_dev, df_test = splitData(df, balanceTrain=True)
    elif args.task == "regression":
        df_train, df_dev, df_test = splitData(df, balanceTrain=False)
    else:
        logger.warning("Task Type Error!")
    logger.info("Using Data:{}".format(df_train.shape[0]))


    # Load the BERT tokenizer.
    logger.info('Loading BERT tokenizer...')
    if args.model_kind == 'bert':
        try:
            tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer, do_lower_case=args.do_lower_case)
        except:
            logger.warning("Tokenizer loading failed")
        model = BertForSequenceClassification.from_pretrained(args.model).to(device)
    elif args.model_kind == 'distilbert':
        try:
            tokenizer = DistilBertTokenizerFast.from_pretrained(args.tokenizer, do_lower_case=args.do_lower_case)
        except:
            logger.warning("Tokenizer loading failed")
        model = DistilBertForSequenceClassification.from_pretrained(args.model).to(device)
        
    word_embeddings = get_word_embeddings(bert_model, input_ids, attention_masks, args.train_batch_size, initial=True).to('cpu')
    logging.debug(word_embeddings.size())
    
    get_word_rating(bert_model, input_ids, word_embeddings, attention_masks, tokenizer, args.gold_word, device)
