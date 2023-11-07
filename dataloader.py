import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from datasets import load_dataset

#CUDA for data loading
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#define dataset
#define dataset
class MyDataset():
    def __init__(self, src, tgt):
        self.src = src.to(device)
        self.tgt = tgt.to(device)
    
    def __len__(self):
        assert len(self.src) == len(self.tgt), 'length of source sentence is not equal to the length of target sentence'
        return len(self.src)

    def __getitem__(self, idx):
        return torch.LongTensor(np.array(self.src[idx])), torch.LongTensor(np.array(self.tgt[idx]))
    

def pad_collate_fn(data):
    eng, deu = list(zip(*data))
    eng = nn.utils.rnn.pad_sequence(eng, batch_first = True, padding_value = 3)
    deu = nn.utils.rnn.pad_sequence(deu, batch_first = True, padding_value = 3)
    
    return [torch.tensor(eng), torch.tensor(deu)]


#prepare dataset to be loaded by dataloader in batches (pack into dictionaries)
datasets =  {'train': MyDataset(preprocessed_d['train'][0],preprocessed_d['train'][1]),
             'val': MyDataset(preprocessed_d['val'][0],preprocessed_d['val'][1]),
             'test': MyDataset(preprocessed_d['test'][0],preprocessed_d['test'][1])}

#prepare dataloader
dataloader = {
    'train': DataLoader(datasets['train'], batch_size= 8, shuffle = True, collate_fn= pad_collate_fn),
    'val': DataLoader(datasets['val'], batch_size= 8, shuffle = False, collate_fn= pad_collate_fn),
    'test': DataLoader(datasets['test'], batch_size= 8, shuffle = False, collate_fn= pad_collate_fn)}


