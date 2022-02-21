import torch
import torch.nn as nn

class AASequenceModel(nn.Module):
    '''
    CNN Model
    Implementation of the "CNN model" architecture 
    applied to the one-hot representation of the data
    Model architecture described in the paper

    Parameters
    ----------
    conv_depth: int
        Number of hidden convloutional layers
    lin_depth: int
        Number of hidden Linear layers for classifying each AA
        
    '''

    def __init__(self, conv_depth=4, lin_depth=3):
        super(AASequenceModel, self).__init__()
        self.conv_depth = conv_depth
        self.lin_depth = lin_depth
        self.conv1 = nn.Conv1d(20,128,21, padding='same')
        self.convHidden = nn.ModuleList([nn.Conv1d(128,128,21, padding='same') for i in range(self.conv_depth)])
        self.reluHidden = nn.ModuleList([nn.ELU() for i in range(self.conv_depth)])
        self.linLayers = nn.ModuleList([nn.Linear(128, 128) for i in range(self.lin_depth)])
        self.reluLin = nn.ModuleList([nn.ELU() for i in range(self.lin_depth)])
        self.linLast = nn.Linear(128,1)
        self.negCnt = 0
        self.allCnt = 0
        
    def dying_relu(self):
        return self.negCnt / self.allCnt

    def forward(self, x):
        '''
        Forward pass of the "CNN model"

        Parameters
        ----------
        x: torch.Tensor
            Tensor representing the one-hot representation of the amino acid sequence
            Shape of the tesnor is N x 20
            Where N is the number of amino acids in the given sequence
            
        Returns
        -------
        torch.Tensor
            The logit estimations for each element of the sequence
        '''
        self.negCnt = 0
        self.allCnt = 0
        y = torch.transpose(x, 1, 2)
        y = self.conv1(y.float())
        for i in range(self.conv_depth):
            y = self.convHidden[i](y)
            self.negCnt += torch.sum(y < 0)
            self.allCnt += torch.numel(y)
            y = self.reluHidden[i](y)
        y = torch.transpose(y, 1, 2)
        for i in range(self.lin_depth):
            y = self.linLayers[i](y)
            self.negCnt += torch.sum(y < 0)
            self.allCnt += torch.numel(y)
            y = self.reluLin[i](y)
        y = self.linLast(y)
        return y
