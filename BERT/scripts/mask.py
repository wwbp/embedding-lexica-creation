import argparse
import logging
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import random
from scipy import stats
import spacy
import tokenizations

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

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
parser.add_argument("--batch_size", type=int, default=32, help="Total batch size for training.")

parser.add_argument("--gold_word", type=str, default=None, help="Gold word rating for evaluation.")

args = parser.parse_args()


def get_word_rating(model, input_ids, attention_masks, text, tokenizer, gold):
    
    logger.info('Getting lexicon')
    
    word2values = {}
    word2freq = {}
    
    if args.do_alignment:
        tokenizer_spacy = spacy.load("../fasttext")
    
    model.to(device)
        
    dataset_ori = TensorDataset(input_ids, attention_masks)
    dataloader_ori = DataLoader(
        dataset_ori, sampler = SequentialSampler(dataset_ori), 
        batch_size = args.batch_size)
    
    prediction_true = []
    logger.info('Getting True Prediction')
    for batch in tqdm(dataloader_ori):
        
        b_input_ids = batch[0].to(device)
        b_attention_masks = batch[1].to(device)
        
        with torch.no_grad():
            logits, _ = model(b_input_ids, b_attention_masks)
            
        if args.task == 'classification':
            logits = logits[:,1]
        
        logits = logits.detach().to('cpu').numpy()
        
        prediction_true.append(logits)
    prediction_true = np.concatenate(prediction_true).flatten()

    logger.info('Getting word values')
    for i in tqdm(range(input_ids.size(0))):
        true = prediction_true[i]
        input_id = input_ids[i]
        attention_mask = attention_masks[i]

        word_count = int(attention_mask.sum().item())
        input_id_matrix = input_id.expand(word_count, -1)
        attention_mask = attention_mask.repeat(word_count, 1)
        attention_mask[torch.arange(word_count), torch.arange(word_count)] = 0
        
        dataset = TensorDataset(input_id_matrix, attention_mask)
        dataloader = DataLoader(
            dataset, sampler = SequentialSampler(dataset), 
            batch_size = args.batch_size)
        
        prediction = []
        for batch in dataloader:
            
            b_input_ids = batch[0].to(device)
            b_attention_masks = batch[1].to(device)

            with torch.no_grad():
                logits, _ = model(b_input_ids, b_attention_masks)
            
            if args.task == 'classification':
                logits = logits[:,1]
                
            logits = logits.detach().to('cpu').numpy()
            prediction.append(logits)
        
        prediction = np.concatenate(prediction).flatten()
        value = true - prediction

        words = tokenizer.convert_ids_to_tokens(input_id)
        
        if args.do_alignment:
            sent_spacy = [token.text.lower() for token in tokenizer_spacy(text[i])]
            _, alignment= tokenizations.get_alignments(words, sent_spacy)
            valid_word = set()
            for index_word, word in enumerate(sent_spacy):
                word_value = 0
                if alignment[index_word]:
                    valid_word.add(word)
                    for index in alignment[index_word]:
                            try:
                                word_value += value[index]
                            except IndexError:
                                logger.info(sent_spacy)
                                logger.info(words)
                                logger.info(value)
                                continue
                if word_value != 0:
                    if word not in word2values:
                        #word2values[word] = [word_value/len(alignment[index_word])]
                        word2values[word] = [word_value]
                    else:
                        #word2values[word].append(word_value/len(alignment[index_word]))
                        word2values[word].append(word_value)
            for token in valid_word:
                if token not in word2freq:
                    word2freq[token] = 1
                else:
                    word2freq[token] += 1
        else:
            for token in set(words):
                if token not in tokenizer.special_tokens_map.values():
                    if token not in word2freq:
                        word2freq[token] = 1
                    else:
                        word2freq[token] += 1
            for index, word in enumerate(words):
                if word not in tokenizer.special_tokens_map.values():
                    if word not in word2values:
                        word2values[word] = [value[index]]
                    else:
                        word2values[word].append(value[index])
             
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

    output_file = os.path.join(args.output_dir, args.dataset+'_'+args.model_kind+'_'+args.task+'_mask.csv')
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
    
    input_ids, attention_masks = get_dataset(df_train.text.values, tokenizer, args.max_seq_length, args.task)
    
    get_word_rating(model, input_ids, attention_masks, df_train.text.values, tokenizer, args.gold_word)
