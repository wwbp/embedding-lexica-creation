import argparse
import logging
import copy

import numpy as np
import pandas as pd
from scipy import stats
from deep_shap import DeepExplainer

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.utils import get_dataset
from utils.bert_utils import get_word_embeddings
from models.cnn import CNN
from models.modeling_bert import BertForSequenceClassification


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()

parser.add_argument("--data", required=True, type=str, help="The input data. Should be .csv file.")
parser.add_argument("--task", required=True, type=str, help="The task for empathy or distress.")
parser.add_argument("--train_batch_size", type=int, default=32, help="Total batch size for training.")
parser.add_argument("--model", required=True, type=str, help="The pretrained Bert model we choose.")
parser.add_argument("--max_seq_length", type=int, default=128,
                    help="The maximum total input sequence length after WordPiece tokenization. "
                    "Sequences longer than this will be truncated, and sequences shorter "
                    "than this will be padded.")
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout for CNN.")
parser.add_argument("--lr", type=float, default=1e-5, help="The initial learning rate for Adam.")
parser.add_argument("--epsilon", type=float, default=1e-8, help="Decay rate for Adam.")
parser.add_argument("--num_warmup_steps", type=int, default=0, help="Steps of training to perform linear learning rate warmup for.")
parser.add_argument("--num_train_epochs", type=int, default=200, help="Total number of training epochs to perform.")
parser.add_argument("--output", type=str, help="The output directory where the lexicon will be written.")
parser.add_argument("--tokenizer", type=str, help="Dir to tokenizer for prediction.")
parser.add_argument("--early_stop", type=bool, default=False, help="Whether set early stopping based on F-score.")
parser.add_argument("--patience", type=int, default=7, help="patience for early stopping.")
parser.add_argument("--delta", type=float, default=0, help="delta for early stopping.")
parser.add_argument("--gold_word", type=str, default=None, help="Gold word rating for evaluation.")

args = parser.parse_args() 


def get_word_rating(model, input_ids, word_embeddings, attention_masks, tokenizer ,gold):
    
    logging.info('Getting lexicon')
    
    logging.info('Getting Shapley values')
    explainer = DeepExplainer(model, {'inputs_embeds':word_embeddings[:50]})
    shap_values = torch.from_numpy(explainer.shap_values({'inputs_embeds':word_embeddings}))[0].mean(axis=-1)
    logging.info("Calculated done!")
    
    word2values = {}
    exclude = ['[CLS]', '[SEP]', '[PAD]']
    
    input_ids = torch.masked_select(input_ids, attention_masks.bool())
    words = tokenizer.convert_ids_to_tokens(input_ids)
    shap_values = torch.masked_select(shap_values, attention_masks.bool())
    
    for index, word in enumerate(words):
        if word not in exclude:
            if word not in word2values:
                word2values[word] = [shap_values[index]]
            else:
                word2values[word].append(shap_values[index])
    
    lexicon = {'Word':[], 'Value': []}
    for word in word2values:
        lexicon['Word'].append(word)
        lexicon['Value'].append(np.mean(word2values[word]))
    
    lexicon_df = pd.DataFrame.from_dict(lexicon).sort_values(by='Value')
    
    if gold is not None:
        gold = pd.read_csv(gold, index_col=0)
        gold = gold[['Word', args.task+'.Mean.Sum']]
        
        merge_df = pd.merge(lexicon_df, gold, how='inner', on=['Word'])
        
        pearson, _ = stats.pearsonr(merge_df['Value'], merge_df[args.task+'.Mean.Sum'])
        logging.info("Pearson for word is %f" % pearson)
    
    logging.info('Writing to %s' % args.output)
    
    lexicon_df.to_csv(args.output)
    
    logging.info('Done!')
        

if __name__=="__main__":
    
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        logging.info('There are %d GPU(s) available.' % torch.cuda.device_count())
        logging.info('We will use the GPU: %s' % str(torch.cuda.get_device_name(0)))

    else:
        logging.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        
    corpus = pd.read_csv(args.data)
    
    data = corpus.essay.values
    values = corpus[args.task].values
    
    # Load the BERT tokenizer.
    logging.info('Loading BERT tokenizer...')
    try:
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    except:
        logging.warning("Tokenizer loading failed")
    
    bert_model = BertForSequenceClassification.from_pretrained(args.model).to(device)
    
    input_ids, attention_masks, values = get_dataset(data, values, tokenizer, args.max_seq_length)
    word_embeddings = get_word_embeddings(bert_model, input_ids, attention_masks, args.train_batch_size, True).to('cpu')
    logging.debug(word_embeddings.size())
    
    bert_model.to('cpu')
    
    get_word_rating(bert_model, input_ids, word_embeddings, attention_masks, tokenizer, args.gold_word)
