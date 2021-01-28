import argparse
import os
import logging
import time

import random
import numpy as np
from scipy import stats

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizerFast, DistilBertTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.preprocess import getData, splitData, balanceData
from utils.utils import format_time, prepare_data
from models.modeling_bert import BertForSequenceClassification
from models.modeling_distilbert import DistilBertForSequenceClassification


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
    parser.add_argument("--test_set", action="store_true", help="Whether pick test set from the whole daata.")
    
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

    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument("--lr", type=float, default=1e-5, help="The initial learning rate for Adam.")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Decay rate for Adam.")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Steps of training to perform linear learning rate warmup for.")
    parser.add_argument("--early_stop", action="store_true", help="Whether set early stopping based on F-score.")
    parser.add_argument("--patience", type=int, default=7, help="patience for early stopping.")
    parser.add_argument("--delta", type=float, default=0, help="delta for early stopping.")

    args = parser.parse_args()
    
    return args


def run_train(device: torch.device, args):

    df = getData(args.dataFolder, args.dataset, args.task)
    if args.task == "classification":
        df_train, df_dev, df_test = splitData(df, balanceTrain=True)
    elif args.task == "regression":
        df_train, df_dev, df_test = splitData(df, balanceTrain=False)
    else:
        logger.warning("Task Type Error!")

    # Load the BERT tokenizer.
    logger.info('Loading BERT tokenizer...')
    if args.model_kind == "bert":
        tokenizer = BertTokenizerFast.from_pretrained(args.model, do_lower_case=args.do_lower_case)
    elif args.model_kind == "distilbert":
        tokenizer = DistilBertTokenizerFast.from_pretrained(args.model, do_lower_case=args.do_lower_case)
    else:
        logger.error("Model kind not supported!")
    
    dataset_train = prepare_data(df_train, tokenizer, args.max_seq_length, args.task)
    dataset_dev = prepare_data(df_dev, tokenizer, args.max_seq_length, args.task)

    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top.
    if args.model_kind == "bert": 
        model = BertForSequenceClassification.from_pretrained(
            args.model,
            num_labels = 2 if args.task == 'classification' else 1, # Set 1 to do regression.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = True, # Whether the model returns all hidden-states.
            )
    elif args.model_kind == "distilbert":
        model = DistilBertForSequenceClassification.from_pretrained(
            args.model,
            num_labels = 2 if args.task == 'classification' else 1, # Set 1 to do regression.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = True, # Whether the model returns all hidden-states.
            )

    model.to(device)
    
    optimizer = AdamW(model.parameters(),
                lr = args.lr, # args.learning_rate - default is 5e-5
                eps = args.epsilon # args.adam_epsilon  - default is 1e-8.
                )

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
    
    total_steps = len(train_dataloader) * args.num_train_epochs
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=total_steps)
    
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
                                    labels=b_labels)

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
        
        # Calculate the Pearson Correlation
        if args.task == 'classification':
            metric = (np.concatenate(prediction_train).flatten() == np.concatenate(true_values_train)).sum()/df_train.shape[0]
        else:
            metric, _ = stats.pearsonr(np.concatenate(prediction_train).flatten(), np.concatenate(true_values_train))             
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        logger.info("  Average training loss: {0:.2f}".format(avg_train_loss))
        if args.task == 'classification':
            logger.info("  Accuracy: {0:.3f}".format(metric))
        else:
            logger.info("  Pearson Correlation: {0:.3f}".format(metric))
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
                                        labels=b_labels)
            
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
        
        # Calculate the Pearson Correlation
        if args.task == 'classification':
            metric = (np.concatenate(prediction_val).flatten() == np.concatenate(true_values_val)).sum()/df_dev.shape[0]
        else:
            metric, _ = stats.pearsonr(np.concatenate(prediction_val).flatten(), np.concatenate(true_values_val))             
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
    
        logger.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
        if args.task == 'classification':
            logger.info("  Accuracy: {0:.3f}".format(metric))
        else:
            logger.info("  Pearson Correlation: {0:.3f}".format(metric))
        logger.info("  Validation took: {:}".format(validation_time))

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
        
    #Test Progress
    logger.info("Running Test...")

    dataset_test = prepare_data(df_test, tokenizer, args.max_seq_length, args.task)
    logger.info('{:>5,} test samples'.format(len(dataset_test)))
    test_dataloader = DataLoader(
                dataset_test,  # The training samples.
                sampler = RandomSampler(dataset_test), # Select batches randomly
                batch_size = args.eval_batch_size # Trains with this batch size.
                )
    
    t0 = time.time()

    # Put the model in evaluation mode
    model = model_to_save
    model.eval()

    # Tracking variables 
    total_test_loss = 0

    # Evaluate data for one epoch
    prediction_test = []
    true_values_test = []
    for batch in test_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():        

            loss, logits, _ = model(b_input_ids, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
        
        # Accumulate the validation loss.
        total_test_loss += loss.item()

        if args.task == 'classification':
            logits = torch.argmax(logits, dim=1)
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
            
        prediction_test.append(logits)
        true_values_test.append(label_ids)
        
    # Calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(test_dataloader)
    
    # Calculate the Pearson Correlation
    if args.task == 'classification':
        metric = (np.concatenate(prediction_test).flatten() == np.concatenate(true_values_test)).sum()/df_test.shape[0]
    else:
        metric, _ = stats.pearsonr(np.concatenate(prediction_test).flatten(), np.concatenate(true_values_test))             
    
    # Measure how long the validation run took.
    test_time = format_time(time.time() - t0)

    logger.info("  Test Loss: {0:.2f}".format(avg_test_loss))
    if args.task == 'classification':
        logger.info("  Accuracy: {0:.3f}".format(metric))
    else:
        logger.info("  Pearson Correlation: {0:.3f}".format(metric))
    logger.info("  Test took: {:}".format(test_time))
    
    logger.info("Training done!")


def run_predict(device: torch.device, args):
    '''Unfinished'''
    df = getData(args.dataFolder, args.dataset, args.task)
    if args.test_set:
        _, _, df = splitData(df, True)
    else:
        df = balanceData(df)

    # Load the BERT tokenizer.
    logger.info('Loading BERT tokenizer...')
    if args.model_kind == "bert":
        tokenizer = BertTokenizerFast.from_pretrained(args.model, do_lower_case=args.do_lower_case)
    elif args.model_kind == "distilbert":
        tokenizer = DistilBertTokenizerFast.from_pretrained(args.model, do_lower_case=args.do_lower_case)
    else:
        logger.error("Model kind not supported!")
    
    dataset = prepare_data(df, tokenizer, args.max_seq_length, args.task)

    prediction_dataloader = DataLoader(
        dataset, sampler = SequentialSampler(dataset), 
        batch_size = args.predict_batch_size)
    
    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top.
    if args.model_kind == "bert": 
        model = BertForSequenceClassification.from_pretrained(
            args.model,
            num_labels = 2 if args.task == 'classification' else 1, # Set 1 to do regression.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = True, # Whether the model returns all hidden-states.
            )
    elif args.model_kind == "distilbert":
        model = DistilBertForSequenceClassification.from_pretrained(
            args.model,
            num_labels = 2 if args.task == 'classification' else 1, # Set 1 to do regression.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = True, # Whether the model returns all hidden-states.
            )
    
    model.to(device)
    
    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    TP = TN = FN = FP = 0

    # Predict 
    for batch in prediction_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():        

            _, logits, _ = model(b_input_ids, 
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

        if args.task == 'classification':
            logits = torch.argmax(logits, dim=1)
            
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Store predictions and true labels
        TP += ((logits == 1) & (label_ids == 1)).sum().item()
        TN += ((logits == 0) & (label_ids == 0)).sum().item()
        FN += ((logits == 0) & (label_ids == 1)).sum().item()
        FP += ((logits == 1) & (label_ids == 0)).sum().item()
    
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)

    logger.info("  ACC: {:}".format(acc))
    logger.info("  F1: {:}".format(F1))
    

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
    
    if args.do_train:
        run_train(device, args)
    elif args.do_predict:
        run_predict(device, args)
    else:
        Exception("Have to do one of the training or prediction!")
    

if __name__ == "__main__":
    main()
            