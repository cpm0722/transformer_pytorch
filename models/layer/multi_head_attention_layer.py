"""
@author : Hansu Kim(@cpm0722)
@when : 2022-08-21
@github : https://github.com/cpm0722
@homepage : https://cpm0722.github.io
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, h, qkv_fc_layer, fc_layer, dropout_rate=0):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.query_fc_layer = copy.deepcopy(qkv_fc_layer) # (d_embed, d_model)
        self.key_fc_layer = copy.deepcopy(qkv_fc_layer)   # (d_embed, d_model)
        self.value_fc_layer = copy.deepcopy(qkv_fc_layer) # (d_embed, d_model)
        self.fc_layer= fc_layer # (d_model, d_embed)
        self.dropout = nn.Dropout(p=dropout_rate)


    def calculate_attention(self, query, key, value, mask):
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, h, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)
        attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, h, seq_len, seq_len)
        attention_prob = self.dropout(attention_prob)
        out = torch.matmul(attention_prob, value) # (n_batch, h, seq_len, d_k)
        return out


    def forward(self, *args, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed)
        # mask: (n_batch, 1, seq_len)
        # return value: (n_batch, h, seq_len, d_k)
        n_batch = query.size(0)

        def transform(x, fc_layer):
            out = fc_layer(x)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h) # (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)
            return out

        query = transform(query, self.query_fc_layer) # (n_batch, h, seq_len, d_k)
        key = transform(key, self.key_fc_layer)       # (n_batch, h, seq_len, d_k)
        value = transform(value, self.value_fc_layer) # (n_batch, h, seq_len, d_k)

        out = self.calculate_attention(query, key, value, mask) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1, 2) # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.fc_layer(out) # (n_batch, seq_len, d_embed)
        return out
