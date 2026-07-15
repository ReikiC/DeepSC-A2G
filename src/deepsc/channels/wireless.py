# -*- coding: utf-8 -*-
"""Physical channel models (AWGN / Rayleigh / Rician).

Each channel is a *callable* object: ``channel(tx, n_var) -> rx``. The math is
unchanged from the original ``Channels`` class; only the packaging changed
(methods -> ``__call__``) so the channels can live in a registry and be
dispatched by name (see ``deepsc.channels``).

Adding the UAV air-to-ground channel later = add a new callable class here (or
a sibling module) and register it — no call site needs to change.
"""
import math

import torch

from deepsc.config import device


class AWGN:
    """Additive white Gaussian noise channel."""

    def __call__(self, tx, n_var):
        rx = tx + torch.normal(0, n_var, size=tx.shape).to(device)
        return rx


class Rayleigh:
    """Rayleigh fading channel with ZF equalization (channel estimation)."""

    def __call__(self, tx, n_var):
        shape = tx.shape
        H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        tx = torch.matmul(tx.view(shape[0], -1, 2), H)
        rx = AWGN()(tx, n_var)
        # Channel estimation
        rx = torch.matmul(rx, torch.inverse(H)).view(shape)
        return rx


class Rician:
    """Rician fading channel with ZF equalization (K-factor mean)."""

    def __init__(self, K=1):
        self.K = K

    def __call__(self, tx, n_var):
        shape = tx.shape
        K = self.K
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        tx = torch.matmul(tx.view(shape[0], -1, 2), H)
        rx = AWGN()(tx, n_var)
        # Channel estimation
        rx = torch.matmul(rx, torch.inverse(H)).view(shape)
        return rx
