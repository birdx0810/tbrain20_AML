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

def map_unk(sentence, tokens, ids, unk_token_id, special_tokens):
    """
    If id([UNK]) = 100 and ids = [9, 100, 27], then the result mapping would be [token(9), token(100), token(27)].
    len(mapping) = len(ids)

    Returns:
        `mapping`: [str, str, str, ...]
    """

    mapping = []
    cur_sentence_index = 0
    for ids_index, cur_id in enumerate(ids):

        if ids_index < len(mapping):
            continue

        if cur_id != unk_token_id:
            cur_token = tokens[ids_index]
            mapping.append(cur_token)
            # update char index
            if cur_token not in special_tokens:
                cur_sentence_index += sentence[cur_sentence_index:].find(cur_token) + len(cur_token) # here: may be -1?
        else:
            # find next non-unk token
            next_non_unk_index = -1
            for temp_ids_index in range(ids_index+1, len(ids)):
                if ids[temp_ids_index] != unk_token_id:
                    next_non_unk_index = temp_ids_index
                    break

            next_non_unk_token = tokens[next_non_unk_index]
            
            # split unk string
            # TODO: check len(split_result) == len(unk)
            next_sentence_index = sentence[cur_sentence_index:].find(next_non_unk_token)
            if next_sentence_index == -1:
                string_to_split = sentence[cur_sentence_index:].strip(' ').split(' ')
            else:
                next_sentence_index += cur_sentence_index
                string_to_split = sentence[cur_sentence_index:next_sentence_index].strip(' ').split(' ')

            mapping.extend(string_to_split)

            # update char index
            cur_sentence_index = next_sentence_index

    return mapping