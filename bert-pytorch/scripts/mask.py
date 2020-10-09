import argparse
import logging
import copy
from logging import log
import os
from os import sep
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy import stats

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from transformers import BertTokenizer, DistilBertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.utils import get_dataset
from utils.bert_utils import get_word_embeddings
from models.ffn import FFN
from models.modeling_bert import BertForSequenceClassification
from models.modeling_distilbert import DistilBertForSequenceClassification


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()

parser.add_argument("--data", required=True, type=str, help="The input data. Should be .csv file.")
parser.add_argument("--task", required=True, type=str, help="The task for empathy or distress.")
parser.add_argument("--FFN_model", action='store_true', help="Whether use FFN or not.")
parser.add_argument("--model_kind", required=True, type=str)
parser.add_argument("--if_bert_embedding", type=bool, default=False, help="If use bert embedding.")
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


def get_word_rating(bert_model, input_ids, attention_masks, tokenizer, gold, word_embeddings=None, model=None):
    
    logging.info('Getting lexicon')
    
    word2values = {}
    exclude = ['[CLS]', '[SEP]', '[PAD]']
    
    bert_model.to(device)
    
    if word_embeddings is None:
        
        dataset_ori = TensorDataset(input_ids, attention_masks)
        dataloader_ori = DataLoader(
            dataset_ori, sampler = SequentialSampler(dataset_ori), 
            batch_size = args.train_batch_size)
        
        prediction_true = []
        logging.info('Getting True Prediction')
        for batch in tqdm(dataloader_ori):
            
            b_input_ids = batch[0].to(device)
            b_attention_masks = batch[1].to(device)
            
            with torch.no_grad():
                logits, _ = bert_model(b_input_ids, b_attention_masks)
                
            logits = logits.detach().to('cpu').numpy()
            
            prediction_true.append(logits)
        prediction_true = np.concatenate(prediction_true).flatten()

        logging.info('Getting word values')
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
                batch_size = args.train_batch_size)
            
            prediction = []
            for batch in dataloader:
                
                b_input_ids = batch[0].to(device)
                b_attention_masks = batch[1].to(device)

                with torch.no_grad():
                    logits, _ = bert_model(b_input_ids, b_attention_masks)
                
                logits = logits.detach().to('cpu').numpy()
                prediction.append(logits)
            
            prediction = np.concatenate(prediction).flatten()
            value = true - prediction
 
            words = tokenizer.convert_ids_to_tokens(input_id)
            
            for index, word in enumerate(words):
                if word not in exclude:
                    if word not in word2values:
                        word2values[word] = [value[index]]
                    else:
                        word2values[word].append(value[index])
            
    else:
        model.to(device)
        
        dataset_ori = TensorDataset(word_embeddings)
        dataloader_ori = DataLoader(
            dataset_ori, sampler = SequentialSampler(dataset_ori), 
            batch_size = args.train_batch_size)
        
        prediction_true = []
        logging.info('Getting True Prediction')
        for batch in tqdm(dataloader_ori):
            
            b_word_embeddings = batch[0].to(device)
            
            with torch.no_grad():
                logits = model(b_word_embeddings)
                
            logits = logits.detach().to('cpu').numpy()
            
            prediction_true.append(logits)
        prediction_true = np.concatenate(prediction_true).flatten()
        
        logging.info('Getting word values')
        for i in tqdm(range(input_ids.size(0))):
            true = prediction_true[i]
            input_id = input_ids[i]
            attention_mask = attention_masks[i]
            
            word_count = int(attention_mask.sum().item())
            input_id_matrix = input_id.expand(word_count, -1)
            attention_mask = attention_mask.repeat(word_count, 1)
            attention_mask[torch.arange(word_count), torch.arange(word_count)] = 0
            
            word_embedding = get_word_embeddings(bert_model, input_id_matrix, attention_mask, 
                                                  args.train_batch_size, 'FFN')
            dataset = TensorDataset(word_embedding)
            dataloader = DataLoader(
                dataset, sampler = SequentialSampler(dataset), 
                batch_size = args.train_batch_size)
            
            prediction = []
            for batch in dataloader:
                
                b_word_embeddings = batch[0].to(device)
                
                with torch.no_grad():
                    logits = model(b_word_embeddings)
                
                logits = logits.detach().to('cpu').numpy()
                
                prediction.append(logits)
            
            prediction = np.concatenate(prediction)
            value = true - prediction
        
            words = tokenizer.convert_ids_to_tokens(input_id)
            
            for index, word in enumerate(words):
                if word not in exclude:
                    if word not in word2values:
                        word2values[word] = [value[index]]
                    else:
                        word2values[word].append(value[index])
             
    lexicon = {'Word':[], 'Value': []}
    for word in word2values:
        lexicon['Word'].append(word)
        lexicon['Value'].append(np.mean(word2values[word]))
    
    lexicon_df = pd.DataFrame.from_dict(lexicon).sort_values(by='Value')
    
    if gold is not None:
        gold = pd.read_csv(gold, sep='\t', header=None)
        gold.columns = ['Word', 'Score', 'Std']
        gold = gold[['Word', 'Score']]
        #gold = gold[['Word', args.task+'.Mean.Sum']]
        
        merge_df = pd.merge(lexicon_df, gold, how='inner', on=['Word'])
        
        logging.info('Writing to %s' % args.output)
    
        merge_df.to_csv(args.output)
        
        pearson, _ = stats.pearsonr(merge_df['Value'], merge_df['Score'])
        logging.info("Pearson for word is %f" % pearson)
    
    logging.info('Done!')
        

if __name__=="__main__":
    
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        logging.info('There are %d GPU(s) available.' % torch.cuda.device_count())
        logging.info('We will use the GPU: %s' % str(torch.cuda.get_device_name(0)))

    else:
        logging.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        
    corpus = pd.read_csv(args.data, sep='\t')
    
    data = corpus.text.values
    values = corpus[args.task].values
    
    logging.info(args.if_bert_embedding)
    logging.debug(os.path.exists(args.gold_word))

    # Load the BERT tokenizer.
    logging.info('Loading BERT tokenizer...')
    if args.model_kind == 'bert':
        try:
            tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
        except:
            logging.warning("Tokenizer loading failed")
        bert_model = BertForSequenceClassification.from_pretrained(args.bert_model).to(device)
    elif args.model_kind == 'distilbert':
        try:
            tokenizer = DistilBertTokenizer.from_pretrained(args.tokenizer)
        except:
            logging.warning("Tokenizer loading failed")
        bert_model = DistilBertForSequenceClassification.from_pretrained(args.bert_model).to(device)
    
    input_ids, attention_masks, values = get_dataset(data, values, tokenizer, args.max_seq_length)
    
    if args.FFN_model:
        word_embeddings = get_word_embeddings(bert_model, input_ids, attention_masks, 
                                                args.train_batch_size, 'FFN')
            
        logging.debug(word_embeddings.size())
        logging.debug(os.path.exists(args.gold_word))
        dataset = TensorDataset(word_embeddings, values)
        
        model = cnn_train(dataset)
        get_word_rating(bert_model, input_ids, attention_masks, tokenizer, args.gold_word, word_embeddings, model)
        
    else:
        get_word_rating(bert_model, input_ids, attention_masks, tokenizer, args.gold_word)
