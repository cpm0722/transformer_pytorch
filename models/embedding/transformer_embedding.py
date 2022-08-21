"""
@author : Hansu Kim(@cpm0722)
@when : 2022-08-21
@github : https://github.com/cpm0722
@homepage : https://cpm0722.github.io
"""

import torch.nn as nn


class TransformerEmbedding(nn.Module):

    def __init__(self, token_embedding, positional_encoding, dropout_rate=0):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = token_embedding
        self.positional_encoding = positional_encoding
        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, x):
        out = self.token_embedding(x)
        out = self.positional_encoding(out)
        out = self.dropout(out)
        return out
