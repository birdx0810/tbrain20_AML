# built-in modules
import os
import math

# 3rd-party modules
import json
import torch
from tqdm import tqdm
from tqdm import trange
import transformers

def encode(tokenizer, text, args):
    # encode news content
    encode_obj = tokenizer(text, add_special_tokens=True, padding='max_length',
                           truncation=True, max_length=args['max_seq_len'])

    input_ids = encode_obj['input_ids']
    attention_mask = encode_obj['attention_mask']
    token_type_ids = encode_obj['token_type_ids']
    position_ids = [ids for ids in range(args['max_seq_len'])]

    return input_ids, attention_mask, token_type_ids, position_ids

# construct dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, all_ids):
        self.all_ids = all_ids

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, index):
        return self.all_ids[index]

    def collate_fn(self, batch):
        # pylint: disable=no-member
        input_ids = torch.LongTensor([data[0] for data in batch])
        attention_mask = torch.FloatTensor([data[1] for data in batch])
        token_type_ids = torch.LongTensor([data[2] for data in batch])
        position_ids = torch.LongTensor([data[3] for data in batch])

        return input_ids, attention_mask, token_type_ids, position_ids

# get dataset
def get_dataset(news, tokenizer, args):
    encode_ids = encode(tokenizer, news, args)
    all_ids = [[ids for ids in encode_ids]]
    dataset = Dataset(all_ids)

    return dataset