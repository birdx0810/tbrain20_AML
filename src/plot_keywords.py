import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from ckiptagger import data_utils, construct_dictionary, WS
import fasttext
import fasttext.util
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import load_data

import logging
import re

def main():
    # ckip tokenizer
    # data_utils.download_data_gdown("../data/ckip/data") # gdrive-ckip
    ws = WS("../data/ckip/data")

    def ckip_tokenizer(text):
        return ws([text])[0]

    def preprocess(text):
        text = text.replace('\t', ' ')
        text = re.sub(' +', '', text)
        return text

    def load_stop_words(path):
        stop_words = []
        with open(path, 'r') as f:
            for line in f:
                stop_words.append(line.strip('\n'))
        return stop_words

    pd.set_option('display.max_colwidth', -1)

    # load data
    csv_path = "../data/tbrain_train_final_0610.csv"
    news_dir = '../data/news'
    stop_words_path = "../data/cn_stopwords.txt"

    df = load_data(news_dir=news_dir, csv_path=csv_path, is_labeling_names=True)
    stop_words = load_stop_words(stop_words_path)
    print(len(stop_words))

    # to tf-idf input
    text_list = df['text'].tolist()
    has_name_list = df['has_name'].tolist()
    print(len(text_list))

    # # limit the number of articles
    # num = 20
    # text_list = text_list[:num]
    # has_name_list = has_name_list[:num]

    # proprocess text
    text_list = [preprocess(article) for article in text_list]

    # tf-idf
    vectorizer = TfidfVectorizer(tokenizer=ckip_tokenizer, lowercase=False, stop_words=stop_words)
    vectorizer = vectorizer.fit(text_list)
    vector = vectorizer.transform(text_list)
    names = np.array(vectorizer.get_feature_names())
    print(len(names))
    print(names[:5])
    
    # load fasttext model
    fasttext.util.download_model('zh', if_exists='ignore')
    ft = fasttext.load_model('cc.zh.300.bin')

    wv_list = []
    label_list = []

    # extract keywords in each article
    k = 5
    for article_index, article in enumerate(text_list):
        article_vector = vectorizer.transform([article])
        sorted_keywords = np.argsort(article_vector.toarray()).flatten()[::-1]
        keywords = names[sorted_keywords][:k]
        # get word vectors of keywords
        for keyword in keywords:
            wv_list.append(ft.get_word_vector(keyword))
            label_list.append(has_name_list[article_index])
        # print('====================')
        # print('article:')
        # print(article)
        # print('keywords:')
        # print(names[sorted_keywords][:k])
        # if article_index == 10:
        #     break

    # t-SNE
    wv_reduced = TSNE(n_components=2).fit_transform(wv_list)
    df_plot = pd.DataFrame()
    df_plot['2d-one'] = [ele[0] for ele in wv_reduced]
    df_plot['2d-two'] = [ele[1] for ele in wv_reduced]
    df_plot['label'] = label_list

    # plot
    plt.figure(figsize=(20,10))
    sns.scatterplot(
        x="2d-one", y="2d-two",
        hue='label',
        data=df_plot,
        legend="full",
        alpha=0.3
    )
    plt.show()
    plt.savefig('../data/stat/temp.png')

    return

if __name__ == '__main__':
    main()
