import argparse
import os
import time
import datetime
from tqdm import tqdm
import numpy as np
from scipy import stats
import random
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", required=True, type=str, 
                    help="The input data dir. Should contain the .csv files"
                    "for the task.")
parser.add_argument("--task", required=True, type=str, 
                    help="The task for empathy or distress")
parser.add_argument("--do_train", type=bool, default=False, help="Whether to run training")
parser.add_argument("--do_predict", type=bool, default=False, 
                    help="Whether to run the model in inference mode on the test set")
parser.add_argument("--model", required=True, type=str, 
                    help="The pretrained Bert model we choose")
parser.add_argument("--do_lower_case", type=bool, default=True,
                    help= "Whether to lower case the input text. Should be True for uncased "
                    "models and False for cased models.")
parser.add_argument("--max_seq_length", type=int, default=128,
                    help="The maximum total input sequence length after WordPiece tokenization. "
                    "Sequences longer than this will be truncated, and sequences shorter "
                    "than this will be padded.")
parser.add_argument("--train_batch_size", type=int, default=32, help="Total batch size for training.")
parser.add_argument("--eval_batch_size", type=int, default=8, help="Total batch size for eval.")
parser.add_argument("--predict_batch_size", type=int, default=8, help="Total batch size for prediction.")
parser.add_argument("--lr", type=float, default=1e-5, help="The initial learning rate for Adam.")
parser.add_argument("--epsilon", type=float, default=1e-8, help="Decay rate for Adam.")
parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
parser.add_argument("--num_warmup_steps", type=int, default=0, help="Steps of training to perform linear learning rate warmup for.")
parser.add_argument("--output_dir", type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--tokenizer", type=str, help="Dir to tokenizer for prediction.")

args = parser.parse_args()


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_dataset(input, values, tokenizer):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in input:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = args.max_seq_length,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                            )
    
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
    
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0).long()
    attention_masks = torch.cat(attention_masks, dim=0).float()
    values = torch.tensor(values).float()
    
    return TensorDataset(input_ids, attention_masks, values)

def run_train():
    train_file = os.path.join(args.data_dir, 'train.csv')

    train_df = pd.read_csv(train_file)

    print('Number of training sentences: {:,}\n'.format(train_df.shape[0]))

    data = train_df.essay.values
    values = train_df[args.task].values

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)

    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained(
        args.model,
        num_labels = 1, # Set 1 to do regression.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        )

    # Tell pytorch to run this model on the GPU.
    model.cuda()
    
    optimizer = AdamW(model.parameters(),
                lr = args.lr, # args.learning_rate - default is 5e-5
                eps = args.epsilon # args.adam_epsilon  - default is 1e-8.
                )
    
    dataset = get_dataset(data, values, tokenizer)

    # Create a 90-10 train-validation split.

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = args.train_batch_size # Trains with this batch size.
                )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = args.eval_batch_size # Evaluate with this batch size.
                )
    
    total_steps = len(train_dataloader) * args.num_train_epochs
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=total_steps)
    
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    total_t0 = time.time()
    
    for epoch_i in tqdm(range(args.num_train_epochs)):

        #Training Progress
        print('Training...')

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

            # Progress update every 40 batches.
            if step % 10 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
    
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

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

            loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)

            total_train_loss += loss.item()
            
            logits = logits.detach().cpu().numpy()
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
        pearson, _ = stats.pearsonr(np.concatenate(prediction_train).flatten(), np.concatenate(true_values_train))             
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Pearson Correlation: {0:.2f}".format(pearson))
        print("  Training epoch took: {:}".format(training_time))

        #Validation Progress
        print("Running Validation...")

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

                (loss, logits) = model(b_input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
            
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            prediction_val.append(logits)
            true_values_val.append(label_ids)

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        # Calculate the Pearson Correlation
        pearson, _ = stats.pearsonr(np.concatenate(prediction_val).flatten(), np.concatenate(true_values_val))             

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
    
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Pearson Correlation: {0:.2f}".format(pearson))
        print("  Validation took: {:}".format(validation_time))
    
        
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    
    output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Saving model to %s" % output_dir)
    
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    

def run_predict():
    test_file = os.path.join(args.data_dir, 'test.csv')

    test_df = pd.read_csv(test_file)

    print('Number of test sentences: {:,}\n'.format(test_df.shape[0]))

    data = test_df.essay.values
    values = test_df[args.task].values
    
    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    try:
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    except:
        print("Tokenizer loading failed")
    
    dataset = get_dataset(data, values, tokenizer)
    prediction_dataloader = DataLoader(
        dataset, sampler = SequentialSampler(dataset), 
        batch_size = args.predict_batch_size)
    
    try:
        model = BertForSequenceClassification.from_pretrained(args.model)
    except:
        print("Model loading failed")
    
    model.cuda()
    
    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
    
    pearson, _ = stats.pearsonr(np.concatenate(predictions).flatten(), np.concatenate(true_labels))
    print("  Pearson Correlation: {0:.2f}".format(pearson))

    print('    DONE.')
    

if __name__ == "__main__":

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    if args.do_train:
        run_train()
    elif args.do_predict:
        run_predict()
    else:
        Exception("Have to do one of the training or prediction!")
            
            
        
        
        
        
        
        
        
        