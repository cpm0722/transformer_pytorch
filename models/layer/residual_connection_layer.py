"""
@author : Hansu Kim(@cpm0722)
@when : 2022-08-21
@github : https://github.com/cpm0722
@homepage : https://cpm0722.github.io
"""

import torch.nn as nn


class ResidualConnectionLayer(nn.Module):

    def __init__(self, norm_layer, dropout_rate=0):
        super(ResidualConnectionLayer, self).__init__()
        self.norm_layer = norm_layer
        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, x, sub_layer):
        out = self.norm_layer(x)
        out = sub_layer(out)
        out = self.dropout(out)
        out = out + x
        return out
