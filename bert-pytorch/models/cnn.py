import logging

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, max_seq_length, embedding_size, dropout):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(max_seq_length, 100 ,1)
        self.conv2 = nn.Conv1d(max_seq_length, 100, 2)
        self.conv3 = nn.Conv1d(max_seq_length, 100, 3)
        self.pooling1 = nn.AvgPool1d(embedding_size)
        self.pooling2 = nn.AvgPool1d(embedding_size-1)
        self.pooling3 = nn.AvgPool1d(embedding_size-2)
        self.fc1 = nn.Linear(300, 128)
        self.fc2 = nn.Linear(128,1)
        self.dropout = nn.Dropout(dropout)
        
        self.criterion = nn.MSELoss()
        
    def forward(self, word_embeddings, values=None):
        x = self.dropout(word_embeddings)
        logging.debug(x.size())
        
        conv_1 = torch.relu(self.conv1(x))
        pool_1 = self.pooling1(conv_1)
        logging.debug(conv_1.size())
        logging.debug(pool_1.size())
        pool_drop_1 = self.dropout(pool_1)
        
        conv_2 = torch.relu(self.conv2(x))
        pool_2 = self.pooling2(conv_2)
        logging.debug(conv_2.size())
        logging.debug(pool_2.size())
        pool_drop_2 = self.dropout(pool_2)
        
        conv_3 = torch.relu(self.conv3(x))
        pool_3 = self.pooling3(conv_3)
        logging.debug(conv_3.size())
        logging.debug(pool_3.size())
        pool_drop_3 = self.dropout(pool_3)
        
        feature = torch.cat([pool_drop_1, pool_drop_2, pool_drop_3]).view(word_embeddings.size()[0],-1)
        z = torch.relu(self.fc1(feature))
        z_drop = self.dropout(z)
        
        outputs = self.fc2(z_drop).flatten()
        
        if values is not None:
            loss = self.criterion(outputs, values)
            
            outputs = (loss, outputs)
            logging.info(outputs)
            logging.info(values)
        return outputs
        