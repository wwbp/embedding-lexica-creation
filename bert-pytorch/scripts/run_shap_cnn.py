import argparse
import logging
import copy
import os

import numpy as np
import pandas as pd
from scipy import stats
from shap import DeepExplainer

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.utils import get_dataset
from utils.bert_utils import get_word_embeddings
import utils.feature_extraction as fe
from utils.embedding import Embedding
from models.cnn import CNN
from models.ffn import FFN
from models.modeling_bert import BertForSequenceClassification


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()

parser.add_argument("--data", required=True, type=str, help="The input data. Should be .csv file.")
parser.add_argument("--task", required=True, type=str, help="The task for empathy or distress.")
parser.add_argument("--model", required=True, type=str, help="Use CNN or FFN.")
parser.add_argument("--if_bert_embedding", type=bool, default=False, help="If use bert embedding.")
parser.add_argument("--embedding_path", type=str, help="Fasttext embedding file.")
parser.add_argument("--train_batch_size", type=int, default=32, help="Total batch size for training.")
parser.add_argument("--bert_model", type=str, help="The pretrained Bert model we choose.")
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


def cnn_train(dataset):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset, sampler = RandomSampler(train_dataset), 
        batch_size = args.train_batch_size)
    
    validation_dataloader = DataLoader(
        val_dataset, sampler = SequentialSampler(val_dataset), 
        batch_size = args.train_batch_size)
    
    try:
        if args.model == 'CNN':
            if args.if_bert_embedding:
                model = CNN(args.max_seq_length-1, dataset[0][0].size()[-1], args.dropout)
            else:
                model = CNN(args.max_seq_length, dataset[0][0].size()[-1], args.dropout)
        else:
            model = FFN(dataset[0][0].size(-1), args.dropout)
    except:
        logging.warning("Model loading failed")
    
    model.to(device)
    
    optimizer = AdamW(model.parameters(),
                lr = args.lr, # args.learning_rate - default is 5e-5
                eps = args.epsilon # args.adam_epsilon  - default is 1e-8.
                )
    
    total_steps = len(train_dataloader) * args.num_train_epochs
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=total_steps)
    
    best_pearson = 0
    count_num = 0
    
    for epoch_i in range(args.num_train_epochs):

        #Training Progress
        logging.info('Training...')

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. 
        model.train()

        # For each batch of training data...
        prediction_train = []
        true_values_train = []
        for step, batch in enumerate(train_dataloader):

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: word embeddings
            #   [1]: attention masks
            #   [2]: labels 
            b_word_embeddings = batch[0].to(device)
            b_labels = batch[1].to(device)

            model.zero_grad()        

            loss, logits = model(b_word_embeddings, b_labels)

            total_train_loss += loss.item()*b_word_embeddings.size()[0]
            
            logits = logits.detach().to('cpu').numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            prediction_train.append(logits)
            true_values_train.append(label_ids)

            # Perform a backward pass to calculate the gradients.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataset)
        
        # Calculate the Pearson Correlation
        pearson, _ = stats.pearsonr(np.concatenate(prediction_train).flatten(), np.concatenate(true_values_train))             

        logging.info("  Average training loss: {0:.2f}".format(avg_train_loss))
        logging.info("  Pearson Correlation: {0:.2f}".format(pearson))
        
        logging.info("Running Validation...")
        
        model.eval()
        
        # Tracking variables 
        total_eval_loss = 0
        
        # Evaluate data for one epoch
        prediction_val = []
        true_values_val = []
        for batch in validation_dataloader:
            
            b_word_embeddings = batch[0].to(device)
            b_labels = batch[1].to(device)
            
            with torch.no_grad():
                loss, logits = model(b_word_embeddings, b_labels)
                
            total_eval_loss += loss.item()*b_word_embeddings.size(0)
            
            logits = logits.detach().to('cpu').numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            prediction_val.append(logits)
            true_values_val.append(label_ids)
            
        avg_val_loss = total_eval_loss /len(val_dataset)
        
        pearson, _ = stats.pearsonr(np.concatenate(prediction_val).flatten(), np.concatenate(true_values_val))
        logging.debug(prediction_val)
        logging.debug(true_values_val)
        logging.info("  Average validation loss: {0:.2f}".format(avg_val_loss))
        logging.info("  Pearson Correlation: {0:.2f}".format(pearson))        

        if pearson < best_pearson+args.delta:
            count_num += 1
        else:
            count_num = 0
            best_pearson = pearson
            
            #Save the model
            best_model = copy.deepcopy(model)
        
        if count_num >= args.patience:
            logging.info('Early stopping')
            break
        
    logging.info("Training complete!")
    return best_model


def get_word_rating(model, input_ids, word_embeddings, attention_masks, tokenizer ,gold):
    
    logging.info('Getting lexicon')
    input_ids = input_ids[:,1:]
    attention_masks = attention_masks[:,1:]
    
    logging.info('Getting Shapley values')
    explainer = DeepExplainer(model, word_embeddings[:50])
    shap_values = torch.from_numpy(explainer.shap_values(word_embeddings))
    logging.debug("---------------------")
    logging.debug(shap_values.size())
    logging.info("Calculated done!")
    
    word2values = {}
    exclude = ['[CLS]', '[SEP]', '[PAD]']
    
    if args.model == 'CNN':
        shap_values = shap_values.mean(axis=-1)
        input_ids = torch.masked_select(input_ids, attention_masks.bool())
        words = tokenizer.convert_ids_to_tokens(input_ids)
        shap_values = torch.masked_select(shap_values, attention_masks.bool())
        
        for index, word in enumerate(words):
            if word not in exclude:
                if word not in word2values:
                    word2values[word] = [shap_values[index]]
                else:
                    word2values[word].append(shap_values[index])
    
    elif args.model == 'FFN':
        shap_values = shap_values.view(input_ids.size(0), input_ids.size(1), -1).mean(axis=-1)
        input_ids = torch.masked_select(input_ids, attention_masks.bool())
        words = tokenizer.convert_ids_to_tokens(input_ids)
        shap_values = torch.masked_select(shap_values, attention_masks.bool())
        
        for index, word in enumerate(words):
            if word not in exclude:
                if word not in word2values:
                    word2values[word] = [shap_values[index]]
                else:
                    word2values[word].append(shap_values[index])
    
    '''
    elif args.model == 'FFN':
        for index, sentence in enumerate(input_ids):
            sentence = tokenizer.convert_ids_to_tokens(sentence)
            for word in sentence:
                if word not in exclude:
                    if word not in word2values:
                        word2values[word] = [shap_values[index]]
                    else:
                        word2values[word].append(shap_values[index])
    ''' 
        
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
        
    corpus = pd.read_csv(args.data)
    
    data = corpus.essay.values
    values = corpus[args.task].values
    
    logging.info(args.if_bert_embedding)
    logging.debug(os.path.exists(args.gold_word))
    if args.if_bert_embedding:
        # Load the BERT tokenizer.
        logging.info('Loading BERT tokenizer...')
        try:
            tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
        except:
            logging.warning("Tokenizer loading failed")
        bert_model = BertForSequenceClassification.from_pretrained(args.bert_model).to(device)
        
        input_ids, attention_masks, values = get_dataset(data, values, tokenizer, args.max_seq_length)
        word_embeddings = get_word_embeddings(bert_model, input_ids, attention_masks, 
                                              args.train_batch_size, args.model).to('cpu')
    else:
        embs=embs = Embedding.from_fasttext_vec(path=args.embedding_path,zipped=True,file='crawl-300d-2M.vec')
        values = torch.tensor(values).float()
        if args.model == 'CNN':
            word_embeddings = torch.from_numpy(fe.embedding_matrix(data, embs, args.max_seq_length)).float()
        elif args.model == 'FFN':
            word_embeddings = torch.from_numpy(fe.embedding_centroid(data, embs)).float()
    logging.debug(word_embeddings.size())
    logging.debug(os.path.exists(args.gold_word))
    dataset = TensorDataset(word_embeddings, values)
    
    model = cnn_train(dataset)
    
    model.to('cpu')
    
    get_word_rating(model, input_ids, word_embeddings, attention_masks, tokenizer, args.gold_word)
