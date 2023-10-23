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

partition = {'train':data_train, 'val': data_val, 'test' : data_test}

#perform preprocessing of data
preprocessed = []
for i in partition.keys():
    #eng, deu = preprocess(partition[i]['translation'])
    preprocessed.append((eng, deu))

#prepare dataset to be loaded by dataloader in batches
dataset_train = MyDataset(preprocessed[0][0],preprocessed[0][1])
dataset_val = MyDataset(preprocessed[1][0],preprocessed[1][1])
dataset_test = MyDataset(preprocessed[2][0],preprocessed[2][1])

#dataloader = DataLoader(dataset_val, batch_size= 8, shuffle = True)


