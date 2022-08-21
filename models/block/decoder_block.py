"""
@author : Hansu Kim(@cpm0722)
@when : 2022-08-21
@github : https://github.com/cpm0722
@homepage : https://cpm0722.github.io
"""

import copy
import torch.nn as nn

from models.layer.residual_connection_layer import ResidualConnectionLayer


class DecoderBlock(nn.Module):

    def __init__(self, masked_multi_head_attention_layer, multi_head_attention_layer, position_wise_feed_forward_layer, norm_layer, dropout_rate=0):
        super(DecoderBlock, self).__init__()
        self.masked_multi_head_attention_layer = masked_multi_head_attention_layer
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm_layer), dropout_rate)
        self.multi_head_attention_layer = multi_head_attention_layer
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm_layer), dropout_rate)
        self.position_wise_feed_forward_layer = position_wise_feed_forward_layer
        self.residual3 = ResidualConnectionLayer(copy.deepcopy(norm_layer), dropout_rate)


    def forward(self, x, encoder_out, mask, encoder_mask):
        out = x
        out = self.residual1(out, lambda out: self.masked_multi_head_attention_layer(query=out, key=out, value=out, mask=mask))
        out = self.residual2(out, lambda out: self.multi_head_attention_layer(query=out, key=encoder_out, value=encoder_out, mask=encoder_mask))
        out = self.residual3(out, self.position_wise_feed_forward_layer)
        return out
