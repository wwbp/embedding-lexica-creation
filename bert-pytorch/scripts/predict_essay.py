import os
import logging
import time

import numpy as np
import pandas as pd
from scipy import stats

import torch
from torch.utils.data import DataLoader, SequentialSampler, random_split, TensorDataset

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.utils import format_time, get_dataset


def run_predict(device: torch.device, args):
    test_file = os.path.join(args.data_dir, 'test.csv')

    test_df = pd.read_csv(test_file)

    logging.info('Number of test sentences: {:,}\n'.format(test_df.shape[0]))

    data = test_df.essay.values
    values = test_df[args.task].values
    
    # Load the BERT tokenizer.
    logging.info('Loading BERT tokenizer...')
    try:
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    except:
        logging.warning("Tokenizer loading failed")
    
    dataset = TensorDataset(get_dataset(data, values, tokenizer, args.max_seq_length))
    prediction_dataloader = DataLoader(
        dataset, sampler = SequentialSampler(dataset), 
        batch_size = args.predict_batch_size)
    
    try:
        model = BertForSequenceClassification.from_pretrained(args.model)
    except:
        logging.warning("Model loading failed")
    
    model.to(device)
    
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
        logits = logits.detach().to('cpu').numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
    
    pearson, _ = stats.pearsonr(np.concatenate(predictions).flatten(), np.concatenate(true_labels))
    print("  Pearson Correlation: {0:.2f}".format(pearson))

    print('    DONE.')