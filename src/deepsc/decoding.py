# -*- coding: utf-8 -*-
"""Inference decoding (greedy autoregressive).

Greedy decoding through the DeepSC transceiver for one batch. For better
quality, swap in beam search here.
"""
import torch

from deepsc.config import device
from deepsc.signal import PowerNormalize
from deepsc.models.masking import subsequent_mask
from deepsc.channels import apply_channel


def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol, channel):
    # create src_mask
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    Rx_sig = apply_channel(channel, Tx_sig, n_var)

    memory = model.channel_decoder(Rx_sig)

    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        # create the decode mask
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.dense(dec_output)

        # predict the word
        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)

        # return the max-prob index
        _, next_word = torch.max(prob, dim=-1)
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs
