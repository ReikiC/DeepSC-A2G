# -*- coding: utf-8 -*-
"""The DeepSC semantic transceiver.

Assembles the semantic Encoder/Decoder (generic Transformer blocks) with a
learned channel encoder (d_model -> 256 -> 16) and channel decoder. This is the
model whose ``state_dict`` is saved to / loaded from ``checkpoints/``; the
parameter keys depend only on the submodule names below, so relocating the
classes does not affect checkpoint compatibility.
"""
import torch.nn as nn
import torch.nn.functional as F

from deepsc.models.transformer import Encoder, Decoder


class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2):
        super(ChannelDecoder, self).__init__()

        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)

        self.layernorm = nn.LayerNorm(size1, eps=1e-6)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)

        output = self.layernorm(x1 + x5)

        return output


class DeepSC(nn.Module):
    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, src_max_len,
                 trg_max_len, d_model, num_heads, dff, dropout=0.1):
        super(DeepSC, self).__init__()

        self.encoder = Encoder(num_layers, src_vocab_size, src_max_len,
                               d_model, num_heads, dff, dropout)

        self.channel_encoder = nn.Sequential(nn.Linear(d_model, 256),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(256, 16))

        self.channel_decoder = ChannelDecoder(16, d_model, 512)

        self.decoder = Decoder(num_layers, trg_vocab_size, trg_max_len,
                               d_model, num_heads, dff, dropout)

        self.dense = nn.Linear(d_model, trg_vocab_size)
