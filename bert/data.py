# built-in modules
import os
import math

# 3rd-party modules
import numpy as np
import pandas as pd
import json
import torch
from tqdm import tqdm
from tqdm import trange
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import transformers

# load news and labels
def load_data(data_path, news_path, save_path):
    data_df = pd.read_csv(data_path)
    for index, row in data_df.iterrows():
        news_id = row.loc['news_ID']

        # get news content
        with open(f'{news_path}/{news_id}.txt', 'r') as f:
            content = f.read().strip()
            content = content.replace('\r', '')
            content = content.replace('\n', '')
            content = content.replace('\t', '')
            data_df.at[index, 'content'] = content

    # convert labels format
    data_df['name'] = data_df['name'].apply(
        lambda name: ','.join(json.loads(name.replace("'", '"'))))

    # remove the data which content is empty
    data_df = data_df[data_df['content'] != '']

    # save new csv file have whole news content
    data_df.to_csv(save_path)

    return data_df

# encode news content and label ids
def encode(tokenizer, text, names, args):
    # encode news content
    encode_obj = tokenizer(text, add_special_tokens=True, padding='max_length',
                           truncation=True, max_length=args['max_seq_len'])
    
    input_ids = encode_obj['input_ids']
    attention_mask = encode_obj['attention_mask']
    token_type_ids = encode_obj['token_type_ids']
    position_ids = [ids for ids in range(args['max_seq_len'])]

    # get label ids
    name_ids = [tokenizer(name, add_special_tokens=False)['input_ids']
                for name in names.split(',')]
    label_ids = [0]*len(input_ids)
    
    for name_id in name_ids:
        ngrams = [input_ids[i:i+len(name_id)] for i in range(0, len(input_ids)-2)]
        for index, ngram in enumerate(ngrams):
            if name_id == ngram:
                label_ids[index:index+len(name_id)] = [1]*len(name_id)

    return input_ids, attention_mask, token_type_ids, position_ids, label_ids

# construct dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, all_ids):
        self.all_ids = all_ids
        
    def __len__(self):
        return len(self.all_ids)
    
    def __getitem__(self, index):
        return self.all_ids[index]
    
    def collate_fn(self, batch):
        input_ids = torch.LongTensor([data[0] for data in batch])
        attention_mask = torch.FloatTensor([data[1] for data in batch])
        token_type_ids = torch.LongTensor([data[2] for data in batch])
        position_ids = torch.LongTensor([data[3] for data in batch])
        label_ids = torch.LongTensor([data[4] for data in batch])
        
        return input_ids, attention_mask, token_type_ids, position_ids, label_ids

# get dataset
def get_dataset(data_df, tokenizer, args):
    all_ids = []

    tqdm.pandas(desc='Encode')
    data_df[['content', 'name']].progress_apply(
        lambda row: encode(tokenizer, row['content'], row['name'], args), axis=1
    ).apply(lambda x: all_ids.append([x[0], x[1], x[2], x[3], x[4]]))

    dataset = Dataset(all_ids)

    return dataset