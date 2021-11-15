import argparse
import os
import logging
import time

import random
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import AdamW, get_scheduler, SchedulerType

from utils.utils import format_time, prepare_data


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
    

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataFolder", required=True, type=str, help="The input data dir.")
    parser.add_argument("--dataset", required=True, type=str, help="The dataset we use.")
    parser.add_argument("--output_dir", type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--task", required=True, type=str, help="Classification or regression.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction.")
    
    parser.add_argument("--model_kind", required=True, type=str, help="Bert or distilBert.")
    parser.add_argument("--model", required=True, type=str, help="The pretrained Bert model we choose.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help= "Whether to lower case the input text. Should be True for uncased \
                            models and False for cased models.")
    parser.add_argument("--tokenizer", type=str, help="Dir to tokenizer for prediction.")

    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Total batch size for eval.")
    parser.add_argument("--predict_batch_size", type=int, default=8, help="Total batch size for prediction.")
    parser.add_argument("--no_special_tokens", action="store_true", 
                        help="whether use the embeddings of the special tokens in the classifier.")

    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument("--lr", type=float, default=1e-5, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Steps of training to perform linear learning rate warmup for.")
    parser.add_argument("--early_stop", action="store_true", help="Whether set early stopping based on F-score.")
    parser.add_argument("--patience", type=int, default=7, help="patience for early stopping.")
    parser.add_argument("--delta", type=float, default=0, help="delta for early stopping.")

    parser.add_argument("--use_lr_model", action="store_true", help="whetehr use logistic model for evaluation.")

    args = parser.parse_args()
    
    return args


def run_train(tokenizer, model, device: torch.device, args):

    path = os.path.join(args.dataFolder, args.dataset)
    df_train = pd.read_csv(os.path.join(path, 'train.csv'))
    df_dev = pd.read_csv(os.path.join(path, 'dev.csv'))
    
    dataset_train = prepare_data(df_train, tokenizer, args.max_seq_length, args.task)
    dataset_dev = prepare_data(df_dev, tokenizer, args.max_seq_length, args.task)

    model.to(device)
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    logger.info('{:>5,} training samples'.format(len(dataset_train)))
    logger.info('{:>5,} validation samples'.format(len(dataset_dev)))

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                dataset_train,  # The training samples.
                sampler = RandomSampler(dataset_train), # Select batches randomly
                batch_size = args.train_batch_size # Trains with this batch size.
                )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                dataset_dev, # The validation samples.
                sampler = SequentialSampler(dataset_dev), # Pull out batches sequentially.
                batch_size = args.eval_batch_size # Evaluate with this batch size.
                )
    
    total_steps = len(train_dataloader) * 3
    
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=total_steps,
    )

    best_metric = 0
    count_num = 0
    
    for epoch_i in range(args.num_train_epochs):

        #Training Progress
        logger.info('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. 
        model.train()

        # For each batch of training data...
        prediction_train = []
        true_values_train = []
        for step, batch in enumerate(train_dataloader):

            # Progress update every 10 batches.
            if step % 10 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
    
                # Report progress.
                logger.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()        

            loss, logits, _ = model(b_input_ids,  
                                    attention_mask=b_input_mask, 
                                    labels=b_labels,
                                    no_special_tokens=args.no_special_tokens)

            total_train_loss += loss.item()
            
            if args.task == 'classification':
                logits = torch.argmax(logits, dim=1)
            
            logits = logits.detach().to('cpu').numpy()
            label_ids = b_labels.to('cpu').numpy()
                
            prediction_train.append(logits)
            true_values_train.append(label_ids)

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)
        logger.info("  Average training loss: {0:.2f}".format(avg_train_loss))
        
        prediction_train = np.concatenate(prediction_train)
        true_values_train = np.concatenate(true_values_train)
        
        # Calculate the Pearson Correlation
        if args.task == 'classification':
            acc = accuracy_score(true_values_train, prediction_train)
            f1 = f1_score(true_values_train, prediction_train)
            logger.info('  Accuracy: {0:.3f}, F1: {0:.3f}'.format(acc, f1))
        else:
            pearson, _ = stats.pearsonr(prediction_train.flatten(), true_values_train)
            logger.info("  Pearson Correlation: {0:.3f}".format(pearson))
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        logger.info("  Training epoch took: {:}".format(training_time))

        #Validation Progress
        logger.info("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode
        model.eval()

        # Tracking variables 
        total_eval_loss = 0

        # Evaluate data for one epoch
        prediction_val = []
        true_values_val = []
        for batch in validation_dataloader:
    
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
    
            with torch.no_grad():        

                loss, logits, _ = model(b_input_ids, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels,
                                        no_special_tokens=args.no_special_tokens)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            if args.task == 'classification':
                logits = torch.argmax(logits, dim=1)
            
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
                
            prediction_val.append(logits)
            true_values_val.append(label_ids)
            
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        logger.info("  Validation Loss: {0:.2f}".format(avg_val_loss))

        prediction_val = np.concatenate(prediction_val)
        true_values_val = np.concatenate(true_values_val)

        # Calculate the Pearson Correlation
        if args.task == 'classification':
            acc = accuracy_score(true_values_val, prediction_val)
            f1 = f1_score(true_values_val, prediction_val)
            logger.info('  Accuracy: {0:.3f}, F1: {0:.3f}'.format(acc, f1))
            metric = acc
        else:
            pearson, _ = stats.pearsonr(prediction_val.flatten(), true_values_val)
            logger.info('  Pearson Correlation: {0:.3f}'.format(pearson))
            metric = pearson
            
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        logger.info('  Validation took: {:}'.format(validation_time))

        if metric < best_metric+args.delta:
            count_num += 1
        else:
            count_num = 0
            best_metric = metric
            
            #Save the model
            output_dir = args.output_dir

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            logger.info("Saving model to %s" % output_dir)

            model_to_save = model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        
        if count_num >= args.patience and args.early_stop:
            logger.info('Early stopping')
            break


def run_predict(tokenizer, model, device: torch.device, args):
    
    if ('dialog' not in args.dataset) and ('song' not in args.dataset) and ('friends' not in args.dataset) and ('emobank' not in args.dataset):
        path = os.path.join(args.dataFolder, args.dataset)
        df = pd.read_csv(os.path.join(path, 'test.csv'))
    else:
        path = os.path.join(args.dataFolder, 'test_datasets')
        df = pd.read_csv(os.path.join(path, args.dataset+'.csv'))

    dataset = prepare_data(df, tokenizer, args.max_seq_length, args.task)
    
    prediction_dataloader = DataLoader(
        dataset, sampler = SequentialSampler(dataset), 
        batch_size = args.predict_batch_size)
    
    model.to(device)
    
    # Put model in evaluation mode
    model.eval()

    total_loss = 0
    score = []
    label = []
    # Predict 
    for batch in prediction_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():        

            loss, logits, _ = model(b_input_ids, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    no_special_tokens=args.no_special_tokens)

        if args.task == 'classification':
            logits = torch.softmax(logits, dim=1)[:,1]

        total_loss += loss.item()

        # Move logits and labels to CPU
        score.append(logits.detach().cpu().numpy())
        label.append(b_labels.to('cpu').numpy())
        
    avg_loss = total_loss / len(prediction_dataloader)
    
    score = np.concatenate(score)
    label = np.concatenate(label)

    if args.use_lr_model:
        score = score.reshape(-1,1)
        
        logModel = LogisticRegression()
        modelAcc = np.round(np.mean(cross_val_score(logModel, score, label, cv=5, scoring='accuracy')),3)
        modelF1 = np.round(np.mean(cross_val_score(logModel, score, label, cv=5, scoring='f1')),3)

        logger.info("  ACC: {:}".format(modelAcc))
        logger.info("  F1: {:}".format(modelF1))
    
    else:
        # Calculate the Pearson Correlation
        if args.task == 'classification':
            acc = accuracy_score(label, score)
            f1 = f1_score(label, score)
            logger.info('  Accuracy: {0:.3f}, F1: {0:.3f}'.format(acc, f1))
        else:
            pearson, _ = stats.pearsonr(score.flatten(), label)
            logger.info('  MSE Loss: {0:.3f}, Pearson Correlation: {0:.3f}'.format(avg_loss, pearson))
    

def main():
    args = parse()
    
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

    if args.model_kind == 'roberta':
        from transformers import RobertaTokenizerFast
        from models.roberta import RobertaForSequenceClassification
        tokenizer = RobertaTokenizerFast.from_pretrained(args.model)
        model = RobertaForSequenceClassification.from_pretrained(
            args.model,
            num_labels = 2 if args.task == 'classification' else 1, # Set 1 to do regression.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = True, # Whether the model returns all hidden-states.
            return_dict=False,)
    elif args.model_kind == 'distilbert':
        from transformers import DistilBertTokenizerFast
        from models.distilbert import DistilBertForSequenceClassification
        tokenizer = DistilBertTokenizerFast.from_pretrained(args.model)
        model = DistilBertForSequenceClassification.from_pretrained(
            args.model,
            num_labels = 2 if args.task == 'classification' else 1, # Set 1 to do regression.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = True, # Whether the model returns all hidden-states.
            return_dict=False,)
    elif args.model_kind == 'bert':
        from transformers import BertTokenizerFast
        from models.bert import BertForSequenceClassification
        tokenizer = BertTokenizerFast.from_pretrained(args.model)
        model = BertForSequenceClassification.from_pretrained(
            args.model,
            num_labels = 2 if args.task == 'classification' else 1, # Set 1 to do regression.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = True, # Whether the model returns all hidden-states.
            return_dict=False,)
    
    if args.do_train:
        run_train(tokenizer, model, device, args)
    elif args.do_predict:
        run_predict(tokenizer, model, device, args)
    else:
        Exception("Have to do one of the training or prediction!")
    

if __name__ == "__main__":
    main()
            