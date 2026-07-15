# -*- coding: utf-8 -*-
"""
Quick BLEU-vs-SNR preview on a subset of the test set.
Reuses the exact same greedy_decode + BleuScore path as evaluate.py,
but evaluates only the first N test sentences so you get the curve
shape in minutes instead of ~2h.

Full-fidelity numbers: run evaluate.py (whole test set) in the background.
"""
import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from deepsc.config import DATA_DIR, device
from deepsc.data import EurDataset, collate_data, SeqtoText
from deepsc.models import DeepSC
from deepsc.signal import SNR_to_noise
from deepsc.metrics import BleuScore
from deepsc.decoding import greedy_decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
    parser.add_argument('--channel', default='Rayleigh', type=str)
    parser.add_argument('--num-sentences', default=2000, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--d-model', default=128, type=int)
    parser.add_argument('--dff', default=512, type=int)
    parser.add_argument('--num-layers', default=4, type=int)
    parser.add_argument('--num-heads', default=8, type=int)
    args = parser.parse_args()

    SNR = [0, 3, 6, 9, 12, 15, 18]

    vocab = json.load(open(DATA_DIR + 'europarl/vocab.json', 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    net = DeepSC(args.num_layers, num_vocab, num_vocab,
                 num_vocab, num_vocab, args.d_model, args.num_heads,
                 args.dff, 0.1).to(device)

    # pick the max-epoch checkpoint (same logic as evaluate.py)
    model_paths = []
    for fn in os.listdir(args.checkpoint_path):
        if not fn.endswith('.pth'):
            continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])
        model_paths.append((os.path.join(args.checkpoint_path, fn), idx))
    model_paths.sort(key=lambda x: x[1])
    model_path, ep = model_paths[-1]
    net.load_state_dict(torch.load(model_path, map_location=device))
    print('loaded checkpoint:', os.path.basename(model_path), '(epoch {})'.format(ep), flush=True)

    test_eur = EurDataset('test')
    n = min(args.num_sentences, len(test_eur))
    sub = Subset(test_eur, list(range(n)))
    test_iterator = DataLoader(sub, batch_size=args.batch_size, num_workers=0,
                               collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, end_idx)
    bleu_1 = BleuScore(1, 0, 0, 0)

    net.eval()
    curve = []
    with torch.no_grad():
        for snr in SNR:
            noise_std = SNR_to_noise(snr)
            word, target_word = [], []
            for sents in test_iterator:
                sents = sents.to(device)
                out = greedy_decode(net, sents, noise_std, 30, pad_idx,
                                    start_idx, args.channel)

                pred = list(map(StoT.sequence_to_text, out.cpu().numpy().tolist()))
                tgt = list(map(StoT.sequence_to_text, sents.cpu().numpy().tolist()))
                word += pred
                target_word += tgt

            scores = bleu_1.compute_blue_score(target_word, word)
            mean_bleu = float(np.mean(scores))
            curve.append(mean_bleu)
            print('SNR {:>2} dB | BLEU-1 = {:.4f}'.format(snr, mean_bleu), flush=True)

    print('\n=== curve (BLEU-1 vs SNR) on {} test sentences ==='.format(n))
    print('SNR  :', SNR)
    print('BLEU :', [round(c, 4) for c in curve])


if __name__ == '__main__':
    main()
