import pandas as pd
import sentencepiece as spm
import sentencepiece as spm
import time
import numpy as np



def dictionary_to_list(pairs_of_sent):
    
    #create an empty list for de, en sentences
    de_en = []

    #assign each sentence in the dictionary into the list
    for i in range(len(pairs_of_sent)):
        sent = pairs_of_sent[i]['de']
        sent2 = pairs_of_sent[i]['en']
        de_en.append(sent)
        de_en.append(sent2)
    
    return de_en


def initialize_tokenizer():
    spm.SentencePieceTrainer.train('--input=iwslt2017_en_de --model_prefix=m_bpe --vocab_size=5000 --model_type=bpe --pad_id=3, --normalization_rule_name=nfkc_cf')
    bpe = spm.SentencePieceProcessor()

    return bpe


def tokenize(ls_of_sentences, bos_eos = False):
    #loading the processor
    bpe = initialize_tokenizer()
    bpe.load('m_bpe.model')

    #enabling the <BOS> and <EOS> tokens as default index
    if bos_eos == True:
        bpe.SetEncodeExtraOptions('bos:eos')
    
    assert bpe.decode_pieces(bpe.encode_as_pieces('Thank you so much!')) == 'Thank you so much!'.lower(), 'decoded sentence != original sentence'

    #tokenization
    for i in range(len(ls_of_sentences)):
        ls_of_sentences[i] = bpe.encode(ls_of_sentences[i])

    return ls_of_sentences

'''
def max_len(ls_of_sentences):
    ''''''return the maximum length of encoded sentence''''''
    max = 0
    for i in ls_of_sentences:
        if len(i) > max:
            max = len(i) #108
    return max
    
    
def split_en_de(ls_of_sentences):
    en = []
    de = []
    for i in range(len(ls_of_sentences)):
        if i % 2 == 0:
            en.append(ls_of_sentences[i])
        else:
            de.append(ls_of_sentences[i])
    
    return en, de
'''               

def preprocess(data):
    ls = dictionary_to_list(data)
    tokenized = tokenize(ls)

    return tokenized

#load datasets(train, validation, test)
data_train = load_dataset('iwslt2017', 'iwslt2017-en-de', split = 'train')
data_val = load_dataset('iwslt2017', 'iwslt2017-en-de', split = 'validation')
data_test = load_dataset('iwslt2017', 'iwslt2017-en-de', split = 'test')

partition = {'train':data_train, 'val': data_val, 'test' : data_test}

#perform preprocessing of data
preprocessed = []
for i in partition.keys():
    eng = tokenize(dictionary_to_list(partition[i]['translation'][0::2]), bos_eos = False)
    deu = tokenize(dictionary_to_list(partition[i]['translation'][1::2]), bos_eos = True)
    preprocessed.append((eng, deu))
    
preprocessed_d = {}
preprocessed_d['train'] = preprocessed[0]
preprocessed_d['val'] = preprocessed[1]
preprocessed_d['test'] = preprocessed[2]