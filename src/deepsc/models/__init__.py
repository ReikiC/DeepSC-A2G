# -*- coding: utf-8 -*-
"""Model components for DeepSC (transceiver, Transformer blocks, MINE)."""
import torch.nn as nn

from deepsc.models.transformer import (
    Encoder,
    Decoder,
    PositionalEncoding,
    MultiHeadedAttention,
    PositionwiseFeedForward,
    EncoderLayer,
    DecoderLayer,
)
from deepsc.models.transceiver import DeepSC, ChannelDecoder
from deepsc.models.mutual_info import Mine
from deepsc.models.masking import create_masks, subsequent_mask

__all__ = [
    "DeepSC", "ChannelDecoder", "Mine",
    "Encoder", "Decoder", "PositionalEncoding", "MultiHeadedAttention",
    "PositionwiseFeedForward", "EncoderLayer", "DecoderLayer",
    "create_masks", "subsequent_mask", "initNetParams",
]


def initNetParams(model):
    """Initialize network parameters with Xavier uniform weights."""
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
