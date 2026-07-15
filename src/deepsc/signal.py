# -*- coding: utf-8 -*-
"""Physical-layer signal helpers (SNR <-> noise, power normalization)."""
import numpy as np
import torch


def SNR_to_noise(snr):
    """Convert an SNR in dB into the corresponding AWGN noise std deviation."""
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std


def PowerNormalize(x):
    """Normalize the average power of a transmitted signal to <= 1."""
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    return x
