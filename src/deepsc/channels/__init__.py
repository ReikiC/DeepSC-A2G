# -*- coding: utf-8 -*-
"""Channel registry and dispatch.

Use ``apply_channel(name, tx, n_var)`` from any training/decoding site instead
of hardcoding an if/elif on the channel name. This collapses the four repeated
``if channel == 'AWGN' / elif 'Rayleigh' / elif 'Rician'`` blocks that used to
live in utils.py into a single dispatch point.

To add a new channel (e.g. the UAV air-to-ground channel), define a callable
class and ``register("A2G", A2G())`` — no existing call site changes.
"""
from deepsc.channels.wireless import AWGN, Rayleigh, Rician

__all__ = ["AWGN", "Rayleigh", "Rician", "apply_channel", "register"]

# name -> callable channel instance
_REGISTRY = {
    "AWGN": AWGN(),
    "Rayleigh": Rayleigh(),
    "Rician": Rician(),
}


def apply_channel(name, tx, n_var):
    """Apply the channel registered under ``name`` to a transmitted signal."""
    try:
        channel = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            "Unknown channel {!r}; choose from {}".format(name, sorted(_REGISTRY))
        )
    return channel(tx, n_var)


def register(name, channel):
    """Register a new channel instance under ``name`` (to extend the set)."""
    _REGISTRY[name] = channel
