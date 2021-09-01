from torch import nn
import torch.nn.functional as F


class NNNet(nn.Module):
    def __init__(self):
        super(NNNet, self).__init__()
        
        self.fc1 = nn.Linear(300, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 2)
        self.drop = nn.Dropout(0.7)        
                
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        # x = self.drop(x)
        x = F.relu(self.fc2(x))
        # x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x