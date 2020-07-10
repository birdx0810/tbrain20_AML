# -*- coding: UTF-8 -*-
import ast
import os

import pandas as pd

import tokenizer

# Path for loading dataset
CSV_PATH = "../data/tbrain_train_final_0610.csv"
NEWS_PATH = os.path.abspath(
    f"../data/news"
)

# Get dataset
def get_dataset(csv_path=None, news_path=None, tokenizer=None):
    df = pd.read_csv(csv_path)

    name_list = df["name"].tolist()
    name_list = [ast.literal_eval(name) for name in name_list]

    news = sorted(os.listdir(news_path))
    news.sort(key=len, reverse=False)
    news = [f"{news_path}/{path}" for path in news if path.endswith(".txt")]

    corpus = []

    for p in news:
        with open(p, "r") as f:
            text = f.readlines()
            text = [line.strip('\n') for line in text]
            corpus.append(' '.join(text))

    tokens = tokenizer.tokenize(corpus)
    labels = tokenizer.labeler(name_list, tokens)
    encoded = tokenizer.encode(tokens)

    dataset = list(zip(encoded, name_list))

    # Drop data without document
    dropped = []
    for data in dataset:
        if data[0] is not "":
            dropped.append(data)

    print(f"# of data: {len(dropped)}")

    return dropped

if __name__ == "__main__":
    t = tokenizer.Tokenizer()
    data = get_dataset(CSV_PATH, NEWS_PATH, t)
