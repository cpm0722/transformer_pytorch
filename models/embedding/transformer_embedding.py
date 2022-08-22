"""
@author : Hansu Kim(@cpm0722)
@when : 2022-08-21
@github : https://github.com/cpm0722
@homepage : https://cpm0722.github.io
"""

import torch.nn as nn


class TransformerEmbedding(nn.Module):

    def __init__(self, token_embed, pos_embed, dr_rate=0):
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)
        self.dropout = nn.Dropout(p=dr_rate)


    def forward(self, x):
        out = x
        out = self.embedding(out)
        out = self.dropout(out)
        return out
