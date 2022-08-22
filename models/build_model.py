"""
@author : Hansu Kim(@cpm0722)
@when : 2022-08-21
@github : https://github.com/cpm0722
@homepage : https://cpm0722.github.io
"""

import torch
import torch.nn as nn

from models.model.transformer import Transformer
from models.model.encoder import Encoder
from models.model.decoder import Decoder
from models.block.encoder_block import EncoderBlock
from models.block.decoder_block import DecoderBlock
from models.layer.multi_head_attention_layer import MultiHeadAttentionLayer
from models.layer.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
from models.embedding.transformer_embedding import TransformerEmbedding
from models.embedding.token_embedding import TokenEmbedding
from models.embedding.positional_encoding import PositionalEncoding


def build_model(src_vocab_size,
                tgt_vocab_size,
                device=torch.device("cpu"),
                max_len = 256,
                d_embed = 512,
                n_layer = 6,
                d_model = 512,
                h = 8,
                d_ff = 2048,
                dr_rate = 0.1,
                norm_eps = 1e-5):
    import copy
    copy = copy.deepcopy

    src_token_embed = TokenEmbedding(
                                     d_embed = d_embed,
                                     vocab_size = src_vocab_size)
    tgt_token_embed = TokenEmbedding(
                                     d_embed = d_embed,
                                     vocab_size = tgt_vocab_size)
    pos_embed = PositionalEncoding(
                                   d_embed = d_embed,
                                   max_len = max_len,
                                   device = device)

    src_embed = TransformerEmbedding(
                                     token_embed = src_token_embed,
                                     pos_embed = copy(pos_embed),
                                     dr_rate = dr_rate)
    tgt_embed = TransformerEmbedding(
                                     token_embed = tgt_token_embed,
                                     pos_embed = copy(pos_embed),
                                     dr_rate = dr_rate)

    attention = MultiHeadAttentionLayer(
                                        d_model = d_model,
                                        h = h,
                                        qkv_fc = nn.Linear(d_embed, d_model),
                                        out_fc = nn.Linear(d_model, d_embed),
                                        dr_rate = dr_rate)
    position_ff = PositionWiseFeedForwardLayer(
                                               fc1 = nn.Linear(d_embed, d_ff),
                                               fc2 = nn.Linear(d_ff, d_embed),
                                               dr_rate = dr_rate)
    norm = nn.LayerNorm(d_embed, eps = norm_eps)

    encoder_block = EncoderBlock(
                                 self_attention = copy(attention),
                                 position_ff = copy(position_ff),
                                 norm = copy(norm),
                                 dr_rate = dr_rate)
    decoder_block = DecoderBlock(
                                 self_attention = copy(attention),
                                 cross_attention = copy(attention),
                                 position_ff = copy(position_ff),
                                 norm = copy(norm),
                                 dr_rate = dr_rate)

    encoder = Encoder(
                      encoder_block = encoder_block,
                      n_layer = n_layer,
                      norm = copy(norm))
    decoder = Decoder(
                      decoder_block = decoder_block,
                      n_layer = n_layer,
                      norm = copy(norm))
    generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(
                        src_embed = src_embed,
                        tgt_embed = tgt_embed,
                        encoder = encoder,
                        decoder = decoder,
                        generator = generator).to(device)
    model.device = device

    return model
