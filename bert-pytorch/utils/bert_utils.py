from tqdm import tqdm
import logging

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset


logger = logging.getLogger(__name__)


def get_word_embeddings(model, input_ids, attention_masks, batch_size, initial=False):
    dataset = TensorDataset(input_ids, attention_masks)    
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)
    
    device = next(model.parameters()).device
    model.eval()
    
    logger.info("Getting word embeddings")
    
    word_embeddings = []
    for batch in tqdm(dataloader):
         
        b_input_ids = batch[0].to(device)
        b_input_masks = batch[1].to(device)
        
        with torch.no_grad():
            #If choose the initial embedding rather than the last layer.
            if initial:
                word_embedding = model.get_word_embeddings(b_input_ids).detach()
            else:
                word_embedding = model(b_input_ids, b_input_masks)[1][-1][:,1:].detach()
                logger.debug(word_embedding.size())
            word_embeddings.append(word_embedding)
            
    return torch.cat(word_embeddings)