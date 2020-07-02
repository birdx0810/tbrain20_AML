# -*- coding: UTF-8 -*-

from shutil import copyfile

import os
import random
import re
import time
import unicodedata
import warnings

from bs4 import BeautifulSoup
from tqdm import tqdm

import pandas as pd
import requests

warnings.filterwarnings("ignore")

def crawl_traverse():
    SENTENCE_LEN = 25
    data_path = "../data/tbrain_train_final_0610.csv"
    df = pd.read_csv(data_path)
    links = df["hyperlink"].to_list()

    PATH_TO_SAVE = '../data/crawled_news'
    if not os.path.exists(PATH_TO_SAVE):
        os.makedirs(PATH_TO_SAVE)

    for idx, link in enumerate(tqdm(links)):
        try:
            response = requests.get(link)
            response.encoding = "UTF-8"

            soup = BeautifulSoup(response.text)
            paragraphs = []
            for child in soup.recursiveChildGenerator():
                name = getattr(child, "name", None)

                # get sentences with character len > 25 and are under <p> tags
                if name == 'p':
                    paragraph = child.getText()

                    # Reprogram whitespaces using regular expression
                    paragraph = re.sub(' +', ' ', paragraph)
                    paragraph = unicodedata.normalize("NFKC", paragraph)

                    if len(paragraph) > SENTENCE_LEN:
                        paragraphs.append(paragraph)

            if len(paragraphs) == 0:
                print('empty')
                copyfile(f"../data/news/{idx}.txt", f"{PATH_TO_SAVE}/{idx+1}.txt")

            else:
                with open(f"{PATH_TO_SAVE}/{idx+1}.txt", "w") as f:
                    for paragraph in paragraphs:
                        f.write(paragraph + '\n')

            time.sleep(random.random() * 1.5 + 0.5)

        except:
            time.sleep(random.random() * 1.5 + 0.5)

    return

if __name__ == "__main__":
    # crawl()
    crawl_traverse()
