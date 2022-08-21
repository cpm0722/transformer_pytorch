"""
@author : Hansu Kim(@cpm0722)
@when : 2022-08-21
@github : https://github.com/cpm0722
@homepage : https://cpm0722.github.io
"""

import copy
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, encoder_block, n_layer, norm_layer):
        super(Encoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(self.n_layer)])
        self.norm_layer = norm_layer


    def forward(self, x, mask):
        out = x
        for layer in self.layers:
            out = layer(out, mask)
        out = self.norm_layer(out)
        return out
