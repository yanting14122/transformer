import pandas as pd
import sentencepiece as spm
import time
import numpy as np
from datasets import load_dataset



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
    templates= '--input={} \
    --pad_id={} \
    --bos_id={} \
    --eos_id={} \
    --unk_id={} \
    --model_prefix={} \
    --vocab_size={} \
    --character_coverage={} \
    --model_type={} \
    --normalization_rule_name={}'


    train_input_file = "iwslt2017_en_de"
    pad_id=0  #<pad> token을 0으로 설정
    vocab_size = 12000 # vocab 사이즈
    prefix = 'm_spm' # 저장될 tokenizer 모델에 붙는 이름
    bos_id=1 #<start> token을 1으로 설정
    eos_id=2 #<end> token을 2으로 설정
    unk_id=3 #<unknown> token을 3으로 설정
    character_coverage = 1.0 # to reduce character set 
    model_type ='bpe' # Choose from unigram (default), bpe, char, or word
    normalization_rule_name = 'nfkc_cf'


    cmd = templates.format(train_input_file,
                pad_id,
                bos_id,
                eos_id,
                unk_id,
                prefix,
                vocab_size,
                character_coverage,
                model_type,
                normalization_rule_name)
    
    spm.SentencePieceTrainer.train(cmd)
    bpe = spm.SentencePieceProcessor()

    return bpe


def tokenize(ls_of_sentences, bos_eos = False):
    #loading the processor
    bpe = initialize_tokenizer()
    bpe.load('m_spm.model')

    #enabling the <BOS> and <EOS> tokens as default index
    if bos_eos == True:
        bpe.SetEncodeExtraOptions('bos:eos')
    
    assert bpe.decode_pieces(bpe.encode_as_pieces('Thank you so much!')) == 'Thank you so much!'.lower(), 'decoded sentence != original sentence'

    #tokenization
    tokenized = []
    for i in range(len(ls_of_sentences)):
        tokenized.append(bpe.encode(ls_of_sentences[i]))

    return tokenized

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
    
preprocessed_d = {'train': preprocessed[0],
                  'val': preprocessed[1],
                  'test': preprocessed[2]}