import argparse
import logging
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy import stats
from deep_shap import Deep as DeepExplainer
import spacy
import tokenizations

import torch
from transformers import BertTokenizerFast, DistilBertTokenizerFast

from utils.preprocess import getData, splitData
from utils.utils import get_dataset
from utils.bert_utils import get_word_embeddings
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
parser.add_argument("--background_size", type=int, default=50,)

parser.add_argument("--gold_word", type=str, default=None, help="Gold word rating for evaluation.")

args = parser.parse_args()


def get_word_rating(model, text, input_ids, word_embeddings, background, tokenizer ,gold):
    
    logging.info('Getting lexicon')
    
    logging.info('Getting Shapley values')
    explainer = DeepExplainer(model, {'inputs_embeds':background})
    
    shap_values = []
    #for batch in tqdm(range(total_batch)):
    for batch in tqdm(range(word_embeddings.size(0))):
        batch_value = explainer.shap_values({'inputs_embeds':word_embeddings[batch:batch+1]})[-1]
        shap_values.append(np.array(batch_value).sum(axis=-1))
    shap_values = np.concatenate(shap_values)
    logging.info("Calculated done!")
    
    word2values = {}
    exclude = ['[CLS]', '[SEP]', '[PAD]']
    
    if args.do_alignment:
        logger.info(os.getcwd())
        tokenizer_spacy = spacy.load("./fasttext")
    
    for index_sent, sent in enumerate(text):
        sent_bert = tokenizer.convert_ids_to_tokens(input_ids[index_sent])
        if args.do_alignment:
            sent_spacy = [token.text.lower() for token in tokenizer_spacy(sent)]
            _, alignment= tokenizations.get_alignments(sent_bert, sent_spacy)
            for index_word, word in enumerate(sent_spacy):
                if word not in exclude:
                    value = 0
                    for index in alignment[index_word]:
                        value += shap_values[index_sent][index]
                    if value != 0:
                        if word not in word2values:
                            word2values[word] = [value/len(alignment[index_word])]
                        else:
                            word2values[word].append(value/len(alignment[index_word]))
        else:
            for index, word in enumerate(sent_bert):
                if word not in exclude:
                    if word not in word2values:
                        word2values[word] = [shap_values[index_sent][index]]
                    else:
                        word2values[word].append(shap_values[index_sent][index])
                elif word == '[PAD]':
                    break
    
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
    
    output_file = os.path.join(args.output_dir, args.dataset+'_'+args.model_kind+'_'+args.task+'_deep.csv')
    logger.info('Writing to %s' % output_file)
    lexicon_df.to_csv(output_file)
    
    logger.info('Done!')
        

if __name__=="__main__":
    
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        logging.info('There are %d GPU(s) available.' % torch.cuda.device_count())
        logging.info('We will use the GPU: %s' % str(torch.cuda.get_device_name(0)))

    else:
        logging.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
           
    #device = torch.device("cpu")
    df = getData(args.dataFolder, args.dataset, args.task)
    df_background = pd.read_csv("./FFN_DeepShap/backgrounds_500/"+args.dataset+"_500_bg.csv")
    assert args.background_size <= 500
    background = df_background.text.values[250-args.background_size/2:250+args.background_size/2]
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
        
    input_ids, attention_masks = get_dataset(df_train.text.values, tokenizer, args.max_seq_length, args.task)
    input_ids_bg, attention_masks_bg = get_dataset(background, tokenizer, args.max_seq_length, args.task)

    word_embeddings = get_word_embeddings(model, input_ids, attention_masks, 32, initial=True)
    word_embeddings_bg = get_word_embeddings(model, input_ids_bg, attention_masks_bg, 32, initial=True)
    
    get_word_rating(model, df_train.text.values, input_ids, word_embeddings, word_embeddings_bg, tokenizer, args.gold_word)
