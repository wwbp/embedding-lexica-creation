import datetime

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


def get_dataset(input, values, tokenizer, max_seq_length):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in input:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_seq_length,           # Pad & truncate all sentences.
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