import pandas as pd

import ast
import os
import sys


def load_data(news_dir, csv_path, is_labeling_names=True):
    df = pd.read_csv(csv_path)
    text_list = []
    name_list = []
    has_name_list = []
    cols_to_return = ['news_ID', 'text', 'name']
    if is_labeling_names:
        cols_to_return.append('has_name')

    for index, row in df.iterrows():
        with open(f'{news_dir}/{row["news_ID"]}.txt', 'r') as f:
            text = f.readlines()
            text = [line.strip('\n') for line in text]
            text = ' '.join(text)
            text_list.append(text)
        
        cur_name_list = ast.literal_eval(row["name"])
        name_list.append(cur_name_list)

        if is_labeling_names:
            if len(cur_name_list) > 0:
                has_name_list.append(1)
            else:
                has_name_list.append(0)

    assert len(text_list) == len(df)

    df['text'] = text_list
    df['name'] = name_list
    if is_labeling_names:
        df['has_name'] = has_name_list

    return df[cols_to_return]


def filter_short_lines(lines, len_limit):
    """
    Read whole text, split tokens by spaces.
    Filter out tokens of lengths < `len_limit`.
    """
    tokens = []
    for line in lines:
        tokens += lines[0].strip('/n').split(' ')
    tokens = [token for token in tokens if len(token) > 25]
        
    return ' '.join(tokens)


def is_mojibake(text):
    """
    Detect if unusual word exists in `text`.

    Args:
    - `text` (str)
    """
    if text.find('\x96') > 0 or text.find('\x99') > 0:
        return True
    return False


def test():
    """
    list names of files containing mojibake
    """

    # config
    data_dir = '../data/cleaned_crawled_news'

    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    files = sorted(files)

    for file_name in files:
        with open(f'{data_dir}/{file_name}', 'r') as f:
            lines = f.readlines()

        lines = ' '.join(lines)
        if is_mojibake(lines):
            print(file_name)

    return

if __name__ == '__main__':
    test()
