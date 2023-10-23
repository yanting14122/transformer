import pandas as pd
import sentencepiece as spm
import sentencepiece as spm
import time
import numpy as np



def dictionary_to_list(pairs_of_sent):
    
    #create an empty list for de, en sentences
    de_en = []

    #assign each sentence in the dictionary into the list
    start = time.time()
    for i in range(len(pairs_of_sent)):
        sent = pairs_of_sent[i]['de']
        sent2 = pairs_of_sent[i]['en']
        de_en.append(sent)
        de_en.append(sent2)
    
    end = time.time()
    print(f'dict_to_list: {end - start}')
    return de_en


def initialize_tokenizer():
    spm.SentencePieceTrainer.train('--input=iwslt2017_en_de --model_prefix=m_bpe --vocab_size=5000 --model_type=bpe --pad_id=3, --normalization_rule_name=nfkc_cf')
    bpe = spm.SentencePieceProcessor()

    return bpe

def tokenize(ls_of_sentences):
    #loading the processor
    bpe = initialize_tokenizer()
    bpe.load('m_bpe.model')

    #enabling the <BOS> and <EOS> tokens as default index
    bpe.SetEncodeExtraOptions('bos:eos')
    assert bpe.decode_pieces(bpe.encode_as_pieces('Thank you so much!')) == 'Thank you so much!'.lower(), 'decoded sentence != original sentence'

    #tokenization
    start = time.time()
    for i in range(len(ls_of_sentences)):
        ls_of_sentences[i] = bpe.encode(ls_of_sentences[i])
    end = time.time()
    print(f'tokenize: {end - start}')
    return ls_of_sentences

def max_len(ls_of_sentences):
    '''return the maximum length of encoded sentence'''
    max = 0
    for i in ls_of_sentences:
        if len(i) > max:
            max = len(i) #108
    return max

def padding(ls_of_sentences, pad_id):
    #padding using token '3' <pad> 
    start = time.time()
    length = max_len(ls_of_sentences)
    for i in ls_of_sentences:
        num_pad = length - len(i)
        ls_pad = np.repeat(3, num_pad)
        i.extend(ls_pad)
    end = time.time()
    print(f'padding: {end - start}')
    return ls_of_sentences
    
    
def split_en_de(ls_of_sentences):
    en = []
    de = []
    start = time.time()
    for i in range(len(ls_of_sentences)):
        if i % 2 == 0:
            en.append(ls_of_sentences[i])
        else:
            de.append(ls_of_sentences[i])
    end = time.time()
    print(f'split_en_de: {end - start}')
    return en, de
               


def preprocess(data):
    ls = dictionary_to_list(data)
    tokenized = tokenize(ls)
    padded = padding(tokenized,3)
    en, de = split_en_de(padded)

    return en, de