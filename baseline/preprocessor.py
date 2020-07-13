# -*- coding: UTF-8 -*-
import ast
import os
import pickle

from sklearn.model_selection import train_test_split
import pandas as pd

import tokenizer
import dataset

# Path for loading dataset
CSV_PATH = "../data/tbrain_train_final_0610.csv"
NEWS_PATH = os.path.abspath(
    f"../data/news"
)

# Get dataset
def create_dataset(csv_path=None, news_path=None, max_seq_len=512, seed=None, tokenizer=None):
    df = pd.read_csv(csv_path)

    name_list = df["name"].tolist()
    news_idx = df["news_ID"].tolist()
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

    cleaned = tokenizer.clean(corpus)
    tokens = tokenizer.tokenize(corpus)
    labels = tokenizer.labeler(name_list, tokens)
    encoded = tokenizer.encode(tokens)

    dataset = list(zip(encoded, labels))

    # Drop data without document
    dropped = []
    for idx, data in enumerate(dataset):
        if data[0] != []:
            if len(data[0]) > max_seq_len:
                dropped.append([data[0][:max_seq_len], data[1][:max_seq_len]])
            else:
                dropped.append(data)

    print(f"# of data: {len(dropped)}")

    df = pd.DataFrame(dropped, columns=["char_enc", "label_enc"])

    train_data, test_data = train_test_split(df, test_size=0.1, random_state=seed)

    train_data.to_csv("./data/train.csv")
    test_data.to_csv("./data/test.csv")

    return train_data, test_data

if __name__ == "__main__":
    t = tokenizer.Tokenizer()
    train_data, test_data = create_dataset(CSV_PATH, NEWS_PATH, 512, 9, t)

    # with open("./data/train.pickle", "wb") as fb:
    #     pickle.dump(train_data, fb)

    # with open("./data/test.pickle", "wb") as fb:
    #     pickle.dump(test_data, fb)


    with open("./data/tokenizer", "wb") as fb:
        pickle.dump(t, fb)

    # train_df = pd.DataFrame(train_data, columns=['document', 'label'])
    # train_df.to_csv("./data/train.csv")

    # test_df = pd.DataFrame(test_data, columns=['document', 'label'])
    # test_df.to_csv("./data/test.csv")
