import argparse
import logging
import os

import pandas as pd
import numpy as np
from scipy import stats
import shap
import spacy
import tokenizations

import torch
from transformers import RobertaTokenizerFast, DistilBertTokenizerFast

from utils.preprocess import getData, splitData
from utils.utils import get_dataset
from models.modeling_roberta import RobertaForSequenceClassification
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

    
def get_word_rating(data, f, tokenizer, gold=None):    
    logger.info('Getting word values')
    explainer = shap.Explainer(f, tokenizer)
    shap_values = explainer(data)

    word2values = {}
    exclude = ['[CLS]', '[SEP]', '[PAD]']
    
    if args.do_alignment:
        tokenizer_spacy = spacy.load("./fasttext")

    for index_sent, sent in enumerate(data):
        sent_bert = tokenizer.tokenize(sent)
        assert len(sent_bert) == len(shap_values.data[index_sent]) - 2
        if args.do_alignment:
            sent_spacy = [token.text.lower() for token in tokenizer_spacy(sent)]
            _, alignment= tokenizations.get_alignments(sent_bert, sent_spacy)
            for index_word, word in enumerate(sent_spacy):
                if word not in exclude:
                    value = 0
                    for index in alignment[index_word]:
                        value += shap_values.values[index_sent][index+1]
                    if value != 0:
                        if word not in word2values:
                            #word2values[word] = [value/len(alignment[index_word])]
                            word2values[word] = [value]
                        else:
                            #word2values[word].append(value/len(alignment[index_word]))
                            word2values[word].append(value)
        else:
            for index_word, word in enumerate(sent_bert):
                if word not in exclude:
                    if word not in word2values:
                        word2values[word] = [shap_values.values[index_sent][index_word+1]]
                    else:
                        word2values[word].append(shap_values.values[index_sent][index_word+1])

    lexicon = {'Word':[], 'Value': [], 'Freq': []}
    for word in word2values:
        lexicon['Word'].append(word)
        lexicon['Value'].append(np.mean(word2values[word]))
        lexicon['Freq'].append(len(word2values[word]))

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
    if args.model_kind == 'roberta':
        try:
            tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer, do_lower_case=args.do_lower_case)
        except:
            logger.warning("Tokenizer loading failed")
        model = RobertaForSequenceClassification.from_pretrained(args.model).to(device)
    elif args.model_kind == 'distilbert':
        try:
            tokenizer = DistilBertTokenizerFast.from_pretrained(args.tokenizer, do_lower_case=args.do_lower_case)
        except:
            logger.warning("Tokenizer loading failed")
        model = DistilBertForSequenceClassification.from_pretrained(args.model).to(device)
    
    def f(x):
        input_ids, attention_masks = get_dataset(x, tokenizer, args.max_seq_length, args.task)
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        outputs = model(input_ids,attention_masks)[0][:, -1].detach().cpu().numpy()
        return outputs
    
    get_word_rating(df_train.text.values, f, tokenizer, args.gold_word)