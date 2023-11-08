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

    def scaled_dot_product(self, Q, K, V):
        attn_scores = torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(self.d_model)
        attention = torch.matmul(F.softmax(attn_scores, dim = -1), V)
        return attention
    
    def split_heads(self, x):
        return x.view(x.size(0), x.size(1), self.d_head, -1).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, max_len, d_head = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, max_len, num_heads * d_head)
    
    def forward(self, Q, K, V):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product(Q, K, V)

        output = self.W_o(self.combine_heads(attn_output))

        return output


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super(Encoder, self).__init__()
        self.self_attn = MultiHeadAttention(512, 8)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = FeedForward(512, 2048)

    def forward(self, x):
        attn_out = self.self_attn(x, x, x)
        attn_out = self.dropout(attn_out)
        sublayer1_out = self.norm1(x + attn_out)
        ff_out = self.feedforward(sublayer1_out)
        ff_out = self.dropout(ff_out)
        sublayer2_out = self.norm2(sublayer1_out + ff_out)
    
        return sublayer2_out
        

