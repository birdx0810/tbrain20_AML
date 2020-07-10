# -*- coding: UTF-8 -*-

import torch
import numpy as np

class AMLDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        super(AMLDataset, self)__init__()
        # Initialize variables
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        return x, y

    def collate_fn(self, batch):

        x = [data[0] for data in batch]
        x = torch.Tensor(x)
        y = torch.nn.utils.rnn.pad_sequence([data[1] for data in batch],
                                            batch_first=True,
                                            padding_value=self.tokenizer.pad_token_idx)

        return x, y

