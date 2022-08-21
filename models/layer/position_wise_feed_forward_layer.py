"""
@author : Hansu Kim(@cpm0722)
@when : 2022-08-21
@github : https://github.com/cpm0722
@homepage : https://cpm0722.github.io
"""

import torch.nn as nn


class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, first_fc_layer, second_fc_layer, dropout_rate=0):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.first_fc_layer = first_fc_layer   # (d_embed, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.second_fc_layer = second_fc_layer # (d_ff, d_embed)


    def forward(self, x):
        out = self.first_fc_layer(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.second_fc_layer(out)
        return out
