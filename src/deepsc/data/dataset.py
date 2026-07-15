# -*- coding: utf-8 -*-
"""Europarl dataset, batching, and token<->text conversion."""
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from deepsc.config import DATA_DIR


class EurDataset(Dataset):
    def __init__(self, split='train'):
        data_dir = DATA_DIR
        with open(data_dir + 'europarl/{}_data.pkl'.format(split), 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        sents = self.data[index]
        return sents

    def __len__(self):
        return len(self.data)


def collate_data(batch):

    batch_size = len(batch)
    max_len = max(map(lambda x: len(x), batch))   # get the max length of sentence in current batch
    sents = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

    for i, sent in enumerate(sort_by_len):
        length = len(sent)
        sents[i, :length] = sent  # padding the questions

    return torch.from_numpy(sents)


class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx

    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return words
