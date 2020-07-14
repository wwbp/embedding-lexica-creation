import logging

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class Classifier(nn.Module):
    def __init__(self, embedding_size, dropout):
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Linear(embedding_size, 1)
        self.dropout = nn.Dropout(dropout)
        
        self.criterion = nn.MSELoss()

        self.apply(self._init_weights)
        
    @staticmethod
    def _init_weights(module, initializer_range:float = 0.02):
        """
        Initialize the weights
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, word_embeddings, values=None):
        #word_embeddings=word_embeddings.permute(0,2,1)
        x = self.dropout(word_embeddings)
        #logger.debug(x.size())
        
        z1 = torch.relu(self.fc1(x))
        z1_drop = self.dropout(z1)
        z2 = torch.relu(self.fc2(z1_drop))
        z2_drop = self.dropout(z2)
        
        outputs = self.fc3(z2_drop)
        
        if values is not None:
            loss = self.criterion(outputs, values.view(-1,1))
            
            outputs = (loss, outputs)
            #logger.info(outputs)
            #logger.info(values)
        return outputs
        
