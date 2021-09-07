import argparse
import logging
import os

import pandas as pd
import numpy as np
from scipy import stats
import shap
import spacy
import tokenizations
import random

import torch

from utils.preprocess import getData, splitData
from utils.utils import get_dataset


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
parser.add_argument("--do_alignment", action="store_true")

parser.add_argument("--max_seq_length", type=int, default=128,
                    help="The maximum total input sequence length after WordPiece tokenization. ")

parser.add_argument("--gold_word", type=str, default=None, help="Gold word rating for evaluation.")

args = parser.parse_args()

    
def get_word_rating(data, f, tokenizer, gold=None):    
    logger.info('Getting word values')
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.explainers.Partition(f, masker)
    shap_values = explainer(data)

    word2values = {}
    word2freq = {}
    
    if args.do_alignment:
        tokenizer_spacy = spacy.load("./fasttext")

    for index_sent, sent in enumerate(data):
        sent_bert = tokenizer.convert_ids_to_tokens(
            tokenizer(
                sent, 
                max_length=args.max_seq_length, 
                padding='max_length', 
                return_tensors = 'pt',
                truncation= True,)["input_ids"][0])
        if args.do_alignment:
            sent_spacy = [token.text.lower() for token in tokenizer_spacy(sent)]
            for token in set(sent_spacy):
                if token not in word2freq:
                    word2freq[token] = 1
                else:
                    word2freq[token] += 1
            _, alignment= tokenizations.get_alignments(sent_bert, sent_spacy)
            for index_word, word in enumerate(sent_spacy):
                value = 0
                for index in alignment[index_word]:
                    try:
                        value += shap_values.values[index_sent][index]
                    except:
                        continue
                if value != 0:
                    if word not in word2values:
                        #word2values[word] = [value/len(alignment[index_word])]
                        word2values[word] = [value]
                    else:
                        #word2values[word].append(value/len(alignment[index_word]))
                        word2values[word].append(value)
        else:
            for token in set(sent_bert):
                if token not in tokenizer.special_tokens_map.values():
                    if token not in word2freq:
                        word2freq[token] = 1
                    else:
                        word2freq[token] += 1
            for index_word, word in enumerate(sent_bert):
                if word not in tokenizer.special_tokens_map.values():
                    if word not in word2values:
                        word2values[word] = [shap_values.values[index_sent][index_word]]
                    else:
                        word2values[word].append(shap_values.values[index_sent][index_word])

    lexicon = {'Word':[], 'Value': [], 'Freq': []}
    for word in word2values:
        lexicon['Word'].append(word)
        lexicon['Value'].append(np.mean(word2values[word]))
        lexicon['Freq'].append(word2freq[word])

    lexicon_df = pd.DataFrame.from_dict(lexicon).sort_values(by='Value')

    if gold is not None:
        gold = pd.read_csv(gold)
        gold.dropna()
        gold.columns = ['Word', 'Score', 'Std']
        gold = gold[['Word', 'Score']]
        #gold = gold[['Word', args.task+'.Mean.Sum']]
        
        merge_df = pd.merge(lexicon_df, gold, how='inner', on=['Word'])
        
        pearson, _ = stats.pearsonr(merge_df['Value'], merge_df['Score'])
        logger.info("Pearson for word is %f" % pearson)

    output_file = os.path.join(args.output_dir, args.dataset+'_'+args.model_kind+'_'+args.task+'_ps.csv')
    logger.info('Writing to %s' % output_file)
    lexicon_df.to_csv(output_file)
    
    logger.info('Done!')


if __name__=="__main__":
    
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        logger.info('There are {} GPU(s) available.'.format(torch.cuda.device_count()))
        logger.info('We will use the GPU: {}'.format(torch.cuda.get_device_name(0)))

    else:
        logger.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    path = os.path.join(args.dataFolder, args.dataset)
    df_train = pd.read_csv(os.path.join(path, 'train.csv'))
    logger.info("Using Data:{}".format(df_train.shape[0]))

    # Load the BERT tokenizer.
    logger.info('Loading BERT tokenizer...')
    if args.model_kind == 'roberta':
        from transformers import RobertaTokenizerFast
        from models.roberta import RobertaForSequenceClassification
        tokenizer = RobertaTokenizerFast.from_pretrained(args.model)
        model = RobertaForSequenceClassification.from_pretrained(args.model).to(device)
    elif args.model_kind == 'distilbert':
        from transformers import DistilBertTokenizerFast
        from models.distilbert import DistilBertForSequenceClassification
        tokenizer = DistilBertTokenizerFast.from_pretrained(args.model)
        model = DistilBertForSequenceClassification.from_pretrained(args.model).to(device)
    
    def f(x):
        input_ids, attention_masks = get_dataset(x, tokenizer, args.max_seq_length, args.task)
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        outputs = model(input_ids,attention_masks)[0]
        scores = torch.nn.Softmax(dim=-1)(outputs)
        val = torch.logit(scores).detach().cpu().numpy()

        return val
    
    get_word_rating(df_train.text.values, f, tokenizer, args.gold_word)