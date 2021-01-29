from typing import *
import datetime
import logging

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, Subset
from torch._utils import _accumulate


logger = logging.getLogger(__name__)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_dataset(input, tokenizer, max_seq_length: int, 
                task: str, values: np.ndarray=None) -> Tuple[torch.Tensor]:
    encoded_dict = tokenizer(
        input.tolist(),                      # Sentence to encode.
        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        max_length = max_seq_length,    # Pad & truncate all sentences.
        pad_to_max_length = True, 
        return_tensors = 'pt',     # Return pytorch tensors.
        truncation= True,
        )

    # Convert the lists into tensors.
    input_ids = encoded_dict['input_ids'].long()
    attention_masks = encoded_dict['attention_mask'].float()
    
    if values is not None:
        if task == "classification":
            values = torch.tensor(values).long()
        elif task == "regression":
            values = torch.tensor(values).float()
        else:
            logger.info("Task type is not permitted")
        
        return input_ids, attention_masks, values
    else:
        return input_ids, attention_masks
    
    
def prepare_data(df: pd.DataFrame, tokenizer, max_seq_length: int, task: str) -> TensorDataset:
    data = df.text.values
    values = df.label.values
    
    input_ids, attention_masks, values, = get_dataset(data, tokenizer, max_seq_length, 
                                                      task, values)
        
    dataset = TensorDataset(input_ids, attention_masks, values)
    
    return dataset


def k_fold_split(dataset, k):
    lengths = [int(len(dataset)/k)]*(k-1)
    lengths.append(len(dataset)-sum(lengths))
    
    indices = torch.randperm(sum(lengths)).tolist()
    subsets = [indices[offset - length:offset] for offset, length in zip(_accumulate(lengths), lengths)]
    
    for i in range(k):
        train_indices = [indice for sub_indices in subsets[0:i]+subsets[i+1:] for indice in sub_indices]
        test_indices = subsets[i]
        yield Subset(dataset, train_indices), Subset(dataset, test_indices)