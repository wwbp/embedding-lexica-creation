import os
import logging
import time

import numpy as np
import pandas as pd
from scipy import stats

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split, TensorDataset

from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.utils import format_time, get_dataset, k_fold_split
from models.modeling_bert import BertForSequenceClassification


def run_train(device: torch.device, args):
    train_file = os.path.join(args.data_dir, 'emobank.csv')

    train_df = pd.read_csv(train_file)

    logging.info('Number of training sentences: {:,}\n'.format(train_df.shape[0]))

    data = train_df.essay.values
    values = train_df[args.task].values

    # Load the BERT tokenizer.
    logging.info('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)

    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained(
        args.model,
        num_labels = 1, # Set 1 to do regression.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = True, # Whether the model returns all hidden-states.
        )

    model.to(device)
    
    optimizer = AdamW(model.parameters(),
                lr = args.lr, # args.learning_rate - default is 5e-5
                eps = args.epsilon # args.adam_epsilon  - default is 1e-8.
                )
    
    input_ids, attention_masks, values = get_dataset(data, values, tokenizer, args.max_seq_length)
    dataset = TensorDataset(input_ids, attention_masks, values)

    if args.k_fold == 0:
        # Create a 90-10 train-validation split.

        # Calculate the number of samples to include in each set.
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        logging.info('{:>5,} training samples'.format(len(train_dataset)))
        logging.info('{:>5,} validation samples'.format(len(val_dataset)))

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
        
        total_t0 = time.time()
        
        best_pearson = 0
        count_num = 0
        
        for epoch_i in range(args.num_train_epochs):

            #Training Progress
            logging.info('Training...')

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
                    logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

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
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)

                total_train_loss += loss.item()
                
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
            pearson, _ = stats.pearsonr(np.concatenate(prediction_train).flatten(), np.concatenate(true_values_train))             
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            logging.info("  Average training loss: {0:.2f}".format(avg_train_loss))
            logging.info("  Pearson Correlation: {0:.2f}".format(pearson))
            logging.info("  Training epoch took: {:}".format(training_time))

            #Validation Progress
            logging.info("Running Validation...")

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
        
            logging.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
            logging.info("  Pearson Correlation: {0:.2f}".format(pearson))
            logging.info("  Validation took: {:}".format(validation_time))

            if pearson < best_pearson+args.delta:
                count_num += 1
            else:
                count_num = 0
                best_pearson = pearson
                
                #Save the model
                output_dir = args.output_dir
        
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                logging.info("Saving model to %s" % output_dir)
        
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            
            if count_num >= args.patience:
                logging.info('Early stopping')
                break
            
    else:
        best_pearsons = []
        for k in range(args.k_fold):
            model = BertForSequenceClassification.from_pretrained(
                args.model,
                num_labels = 1, # Set 1 to do regression.
                output_attentions = False, # Whether the model returns attentions weights.
                output_hidden_states = True, # Whether the model returns all hidden-states.
                )

            model.to(device)
            # Divide the dataset by randomly selecting samples.
            split = k_fold_split(dataset, args.k_fold)
            train_dataset, val_dataset = next(split)

            logging.info('{:>5,} training samples'.format(len(train_dataset)))
            logging.info('{:>5,} validation samples'.format(len(val_dataset)))

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
            
            total_t0 = time.time()
            
            best_pearson = 0
            count_num = 0
            
            for epoch_i in range(args.num_train_epochs):

                #Training Progress
                logging.info('Training...')

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
                        logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

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
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask, 
                                        labels=b_labels)

                    total_train_loss += loss.item()
                    
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
                pearson, _ = stats.pearsonr(np.concatenate(prediction_train).flatten(), np.concatenate(true_values_train))             
                # Measure how long this epoch took.
                training_time = format_time(time.time() - t0)

                logging.info("  Average training loss: {0:.2f}".format(avg_train_loss))
                logging.info("  Pearson Correlation: {0:.2f}".format(pearson))
                logging.info("  Training epoch took: {:}".format(training_time))

                #Validation Progress
                logging.info("Running Validation...")

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
            
                logging.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
                logging.info("  Pearson Correlation: {0:.2f}".format(pearson))
                logging.info("  Validation took: {:}".format(validation_time))

                if pearson < best_pearson+args.delta:
                    count_num += 1
                else:
                    count_num = 0
                    best_pearson = pearson
                
                if count_num >= args.patience:
                    best_pearsons.append(best_pearson)
                    logging.info('Early stopping')
                    break
        
        logging.info("  Mean Pearson Correlation: {0:.2f}".format(sum(best_pearsons)/len(best_pearsons)))
        
    logging.info("Training complete!")
    logging.info("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))