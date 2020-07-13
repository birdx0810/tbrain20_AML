# -*- coding: UTF-8 -*-

import torch
import numpy as np

import sys

class AMLDataset(torch.utils.data.Dataset):
    def __init__(self, config, dataset, tokenizer):
        super(AMLDataset, self).__init__()
        # Initialize variables
        self.config = config
        self.tokenizer = tokenizer

        self.x = [torch.LongTensor(data[0]) for data in dataset]
        self.y = [torch.FloatTensor(data[1]) for data in dataset]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # print(self.x[index].size())
        # print(self.y[index].size())

        x = self.x[index]
        y = self.y[index]

        return x, y

    def collate_fn(self, batch):
        x = torch.nn.utils.rnn.pad_sequence([data[0] for data in batch],
                                            batch_first=True,
                                            padding_value=self.tokenizer.pad_token_idx)
        y = torch.nn.utils.rnn.pad_sequence([data[1] for data in batch],
                                            batch_first=True,
                                            padding_value=self.tokenizer.pad_token_idx)

        return x, y

