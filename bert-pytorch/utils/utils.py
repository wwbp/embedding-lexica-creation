import datetime
import logging

import torch
from torch.utils.data import Subset
from torch._utils import _accumulate


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_dataset(input, values, tokenizer, max_seq_length, task):
    encoded_dict = tokenizer(
                        input.tolist(),                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_seq_length,    # Pad & truncate all sentences.
                        pad_to_max_length = True, 
                        return_tensors = 'pt',     # Return pytorch tensors.
                        )

    # Convert the lists into tensors.
    input_ids = encoded_dict['input_ids'].long()
    attention_masks = encoded_dict['attention_mask'].float()
    
    logging.info(values)
    if '-' in task and task.split('-')[0] == 'classification':
        for i in range(len(values)):
            values[i] = (values[i] == task.split('-')[1])
        values = torch.tensor(values.astype(int)).long()
    else:
        values = torch.tensor(values).float()
    
    return input_ids, attention_masks, values


def k_fold_split(dataset, k):
    lengths = [int(len(dataset)/k)]*(k-1)
    lengths.append(len(dataset)-sum(lengths))
    
    indices = torch.randperm(sum(lengths)).tolist()
    subsets = [indices[offset - length:offset] for offset, length in zip(_accumulate(lengths), lengths)]
    
    for i in range(k):
        train_indices = [indice for sub_indices in subsets[0:i]+subsets[i+1:] for indice in sub_indices]
        test_indices = subsets[i]
        yield Subset(dataset, train_indices), Subset(dataset, test_indices)