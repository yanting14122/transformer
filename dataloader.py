import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from datasets import load_dataset

#CUDA for data loading
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#define dataset
class MyDataset():
    def __init__(self, src, tgt):
        self.src = torch.Tensor(np.array(src))
        self.tgt = torch.Tensor(np.array(tgt))
    
    def __len__(self):
        assert len(self.src) == len(self.tgt), 'length of source sentence is not equal to the length of target sentence'
        return len(self.src)

    def __getitem__(self, idx):
        sample = self.src[idx]
        label = self.tgt[idx]
        return sample, label

#load datasets(train, validation, test)
data_train = load_dataset('iwslt2017', 'iwslt2017-en-de', split = 'train')
data_val = load_dataset('iwslt2017', 'iwslt2017-en-de', split = 'validation')
data_test = load_dataset('iwslt2017', 'iwslt2017-en-de', split = 'test')


#prepare dataset to be loaded by dataloader in batches (pack into dictionaries)
datasets =  {'train': MyDataset(preprocessed['train'][0],preprocessed['train'][1]),
             'val': MyDataset(preprocessed['val'][0],preprocessed['val'][1]),
             'test': MyDataset(preprocessed['test'][0],preprocessed['test'][1])}

#prepare dataloader
dataloader = {
    'train': DataLoader(datasets['train'], batch_size= 8, shuffle = True),
    'val': DataLoader(datasets['val'], batch_size= 8, shuffle = False),
    'test': DataLoader(datasets['test'], batch_size= 8, shuffle = False)}


