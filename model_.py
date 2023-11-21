import torch
import torch.nn as nn
#from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype = torch.float).unsqueeze(1)
        div_term = 10000**(torch.arange(0, d_model ,2)/d_model)

        pe[:, 0::2] = torch.sin(position/div_term)
        pe[:, 1::2] = torch.cos(position/div_term)

        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model))
    
    def forward(self, x):
        out = self.layers(x)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        #linear layer for query, key and value transformation
        self.W_q = nn.Linear(d_model, d_model) 
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product(self, Q, K, V, mask = None):
        attn_scores = torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(self.d_model)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attention = torch.matmul(F.softmax(attn_scores, dim = -1), V)
        return attention
    
    def split_heads(self, x):
        return x.view(x.size(0), x.size(1), self.d_head, -1).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, max_len, d_head = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, max_len, num_heads * d_head)
    
    def forward(self, Q, K, V, mask = None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product(Q, K, V, mask)

        output = self.W_o(self.combine_heads(attn_output))

        return output


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super(Encoder, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = FeedForward(d_model, 2048)

    def forward(self, x, mask):
        attn_out = self.self_attn(x, x, x, mask)
        attn_out = self.dropout(attn_out)
        x = self.norm1(x + attn_out)
        ff_out = self.feedforward(x)
        ff_out = self.dropout(ff_out)
        x = self.norm2(x + ff_out)
    
        return x
    

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super(Decoder, self).__init()
        self.masked_attn = MultiHeadAttention(d_model, num_heads)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = FeedForward(d_model, 2048)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        masked_attn_out = self.masked_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(masked_attn_out))
        self_attn_out = self.self_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(self_attn_out))
        ff_out = self.feedforward(x)
        dec_out = self.norm3(x + self.dropout(ff_out))

        return dec_out