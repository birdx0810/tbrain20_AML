# -*- coding: UTF-8 -*-

import os
import time
import warnings

import re
import os
import unicodedata

from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
import random
import pandas as pd

warnings.filterwarnings("ignore")

def crawl():
    data_path = "data/tbrain_train_final_0610.csv"
    df = pd.read_csv(data_path)
    links = df["hyperlink"].to_list()

    PATH_TO_SAVE = os.path.abspath(
        f"{os.path.abspath(__file__)}/../../data/news/"
    )

    if not os.path.exists(PATH_TO_SAVE):
        os.makedirs(PATH_TO_SAVE)

    for idx, link in enumerate(tqdm(links)):
        while True:
            try:
                response = requests.get(link)

                soup = BeautifulSoup(response.text)
                page = soup.getText()

                # Reprogram whitespaces using regular expression
                prog = re.compile(r"\s+")

                tokens = prog.split(
                    # Encode to UTF-8 and decode whole space characters to halfspace
                    unicodedata.normalize("NFKC", page.strip()))
                output = " ".join(tokens)

                with open(f"{PATH_TO_SAVE}/{idx}.txt", "w") as f:
                    f.write(output)

                time.sleep(random.random() * 1.5 + 0.5)
                break
            except Exception as err:
                time.sleep(random.random() * 1.5 + 0.5)

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

            with open(f"{PATH_TO_SAVE}/{idx}.txt", "w") as f:
                for paragraph in paragraphs:
                    f.write(paragraph + '\n')

            time.sleep(random.random() * 1.5 + 0.5)

        except:
            time.sleep(random.random() * 1.5 + 0.5)

    return

if __name__ == "__main__":
    # crawl()
    crawl_traverse()
