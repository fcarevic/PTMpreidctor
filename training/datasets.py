import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from utils import *
import re

# Total number of amino acids
AA_CNT = 20


class AASequenceDatasetLinear(Dataset):
    """
    Dataset class for the sequence linear model.
    
    This is the dataset for the model which will only take into consideration the amino acid 
    sequence.
    """

    def __init__(self, tsv_file, use_embeddings = False, num_neighbours = 20):

        """
        Parameters
        ----------
        tsv_file : String
            The path to the adequate .tsv file
        num_neighbours : int
            Total number of neighbouring AA to perserve while extracting (does not include targeting AA)
        use_embeddings : boolean
            If set to `True` PESTO embeddings will be used instead of one-hot expansion
      
        Returns
        -------
        torch.Tensor
            The input sequence
        torch.Tesnor
            The mask where values of 1. denote that the input is in use while 0 means that it is masked
        torch.Tensor
            The label
        """
        self.num_neighbours = num_neighbours
        self.use_embeddings = use_embeddings
        self.data = pd.read_csv(tsv_file, sep='\t')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        seq=[]
        mask=[]
        label=[]
        pesto_embedding = []
        
        # Converting string to integer array
        row = self.data.iloc[idx]
        seq = row['seq']
        seq = re.findall(r'\d+', seq)
        seq = list(map(int, seq))
        seq = torch.tensor(seq)

        # Crafting one-hot representation of the sequence of AA
        onehot_seq = np.zeros(seq.shape[0]*AA_CNT)
        onehot_seq[(torch.arange(0,seq.shape[0] * AA_CNT, AA_CNT) + seq).int()] = 1.0
        onehot_seq = onehot_seq.reshape(seq.shape[0], AA_CNT)
        seq = torch.tensor(onehot_seq)

        with torch.no_grad():       
            if self.use_embeddings:
                # Load pesto file
                with open(row['file'],'rb') as file:
                    pesto_embedding = torch.tensor(np.load(file))
               
            # Defining mask for the amino-acids
            mask = torch.ones(seq.shape[0])
            
            # Defining one-sided neighbourhood
            neighbourhood = int(self.num_neighbours/2)
     
            # Adding padding for the boundary AAs in one-hot
            padding = torch.zeros((neighbourhood, seq.shape[1])).double()
            seq = torch.concat([padding,seq])
            seq = torch.concat([seq,padding])
            
            # Adding padding for the boundary AAS in pesto
            if self.use_embeddings:
                padding = torch.zeros((neighbourhood, pesto_embedding.shape[1])).double()
                pesto_embedding = torch.concat([padding,pesto_embedding])
                pesto_embedding = torch.concat([pesto_embedding,padding])

            # Finding indices of relevant AAs (S, T, Y) in one-hot
            # Note that these indices correspond to indices of same AAs
            # the dataset of pesto embeddings
            index_S = (seq[:,aa_to_ord('S')]==1).nonzero()[:,0]
            index_T = (seq[:, aa_to_ord('T')]==1).nonzero()[:,0]
            index_Y = (seq[:, aa_to_ord('Y')]==1).nonzero()[:,0]
            indices = torch.concat([index_S, index_T, index_Y]) 
            
            if self.use_embeddings:
                # Extracting neighbourhoods for each AAs (S, T, Y) in the pesto data
                seq = torch.concat([torch.reshape(pesto_embedding[ind- neighbourhood:ind+neighbourhood+1],(-1,
                                                              pesto_embedding.shape[1] * (self.num_neighbours + 1)                                                   
                                                             )) for ind in indices])
            else :
                # Extracting neighbourhoods for each AAs (S, T, Y) in the one-hot data
                seq = torch.concat([torch.reshape(seq[ind- neighbourhood:ind+neighbourhood+1],(-1,
                                                              seq.shape[1] * (self.num_neighbours + 1)                                                   
                                                             )) for ind in indices])

            positions = row['Position']
            positions = re.findall(r'\d+', positions)
            positions = list(map(int, positions))
            positions = list(map(lambda x : x-1,positions))
            positions = torch.tensor(positions)
            
            # Shifting positions for the lenght of added padding 
            positions = positions + neighbourhood
            
            # Generating labels for each datapoing in seq
            label_mask = torch.isin(indices, positions)
            label = torch.zeros(indices.shape[0])
            label[label_mask]=1

        return seq, mask, label

def cut_and_pad(x, cutlen):
    """
    Cuts and and adds padding to tensor

    This function converts sequences (i.e. tensors) of variable length into ones of uniform length. 
    This is done by cutting sequences which are too long or by adding zero padding to shorter
    sequences in such a way that the end sequence has exactly cutlen elements.

    Parameters
    ----------
    x : torch.Tensor
        The variable length sequence
    cutlen : int
        The number of elements the sequence will have after this function is called

    Returns
    -------
    torch.Tensor
        Uniform length tensor constructed from the input tensor
    """
    temp = x[:cutlen] # cut
    padding = cutlen-len(temp)
    return (
            torch.cat([temp, torch.zeros(padding)]), 
            torch.cat([torch.ones(temp.shape[0]), torch.zeros(padding)])
        )

class AASequenceDataset(Dataset):
    """
    Dataset class for the sequence only model
    
    This is the dataset for the model which will only take into consideration the amino acid 
    sequence.
    """

    def __init__(self, tsv_file, maxlen=None, onehot_input=True, multihot_output=True, equal_size=False, pesto=False):

        """
        Parameters
        ----------
        tsv_file : String
            The path to the adequate .tsv file
        maxlen : int
            The length all sequences will be cut off at or padded to. If `None` then it will be set to
            the length of the longest sequences (no cutting)
        onehot_input : boolean
            If set to `True` all of the input sequence elements will be expaneded to a onehot encoding.
            Otherwise, the input sequence will be encoded by ordinal values
        mutlihot_output : boolean
            If set to `True` the labels will be arrays of ones and zeros where ones denote the PTM sites
            Otherwise, the labels will be encoded by ordinal values

        Returns
        -------
        torch.Tensor
            The input sequence
        torch.Tesnor
            The mask where values of 1. denote that the input is in use while 0 means that it is masked
        torch.Tensor
            The label
        """

        self.data = pd.read_csv(tsv_file, sep='\t')
        self.equal_size = equal_size
        self.onehot_input = onehot_input
        self.multihot_output = multihot_output
        self.pesto = pesto
        if(maxlen == None):
            self.maxlen = np.max(self.data['seq_len'])
        else:
            self.maxlen = maxlen

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        with torch.no_grad():

            
            row = self.data.iloc[idx]
            if(self.pesto):
                with open(row['file'], 'rb') as f:
                    seq = torch.tensor(np.load(f))
            else:
                seq = row['seq']
                seq = re.findall(r'\d+', seq)
                seq = list(map(int, seq))
                seq = torch.tensor(seq)

            if(self.equal_size):
                seq, mask = cut_and_pad(seq, self.maxlen)
            else:
                mask = torch.ones(seq.shape[0])

            if(self.onehot_input and not self.pesto):
                onehot_seq = np.zeros(seq.shape[0]*AA_CNT)
                onehot_seq[(torch.arange(0,seq.shape[0] * AA_CNT, AA_CNT) + seq).int()] = 1.0
                onehot_seq = onehot_seq.reshape(seq.shape[0], AA_CNT)
                seq = onehot_seq

            label = row['Position']
            label = re.findall(r'\d+', label)
            label = list(map(int, label))
            label = list(map(lambda x : x-1,label))

            if(self.multihot_output):
                if(self.equal_size):
                    binlabel = torch.zeros(self.maxlen)
                else:
                    binlabel = torch.zeros(seq.shape[0])
                binlabel[label] = 1.
                label = binlabel
        
        return seq, mask, label

    
    
