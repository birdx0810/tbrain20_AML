# -*- coding: UTF-8 -*-

from shutil import copyfile

import os
import random
import re
import time
import unicodedata
import warnings
import sys

from bs4 import BeautifulSoup
from tqdm import tqdm

import pandas as pd
import requests

from data_utils import filter_short_lines, is_mojibake

warnings.filterwarnings("ignore")

def crawl_url():
    """
    crawl a specified url, and print its content.
    """

    SENTENCE_LEN = 25
    url = 'https://hk.on.cc/hk/bkn/cnt/cnnews/20191226/bkn-20191226091206832-1226_00952_001.html'

    response = requests.get(url)
    print(response.encoding)

    soup = BeautifulSoup(response.text)
    paragraphs = []
    for child in soup.recursiveChildGenerator():
        name = getattr(child, "name", None)

        # get sentences with character len > 25 and are under <p> tags
        if name == 'div' and child.has_attr('class') and child['class'][0] == 'paragraph':
            paragraph = child.getText().encode('latin1').decode('utf-8')

            # Reprogram whitespaces using regular expression
            paragraph = re.sub(' +', ' ', paragraph)
            paragraph = paragraph.replace('\r', '')
            paragraph = paragraph.replace('\t', '')
            paragraph = paragraph.replace('\n', '')
            paragraph = unicodedata.normalize("NFKC", paragraph)

            if len(paragraph) > SENTENCE_LEN:
                paragraphs.append(paragraph)

    print(paragraphs)
    return


def crawl_mojibake():
    """
    get file names containing mojibake
    recrawl those urls corresponding to the files
    """

    # get file names containing mojibake
    data_dir = '../data/cleaned_crawled_news'

    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    files = sorted(files)

    mojibake_indexes = []
    for file_name in files:
        with open(f'{data_dir}/{file_name}', 'r') as f:
            lines = f.readlines()

        lines = ' '.join(lines)
        if is_mojibake(lines):
            ind = file_name.find('.txt')
            mojibake_indexes.append(int(file_name[:ind]))

    print(len(mojibake_indexes))
    print(mojibake_indexes)

    # read csv
    data_path = "../data/tbrain_train_final_0610.csv"
    df = pd.read_csv(data_path)

    # recrawl mojibake urls
    SENTENCE_LEN = 25
    indexes = mojibake_indexes

    PATH_TO_SAVE = '../data/mojibake_news'
    if not os.path.exists(PATH_TO_SAVE):
        os.makedirs(PATH_TO_SAVE)

    for idx in tqdm(indexes):
        try:
            link = df.iloc[idx-1]['hyperlink']
            response = requests.get(link)

            soup = BeautifulSoup(response.text)
            paragraphs = []
            for child in soup.recursiveChildGenerator():
                name = getattr(child, "name", None)

                # get sentences with character len > 25 and are under <p> tags
                if name == 'div' and child.has_attr('class') and child['class'][0] == 'paragraph':
                    paragraph = child.getText().encode('latin1').decode('utf-8')

                    # Reprogram whitespaces using regular expression
                    paragraph = re.sub(' +', ' ', paragraph)
                    paragraph = paragraph.replace('\r', '')
                    paragraph = paragraph.replace('\t', '')
                    paragraph = paragraph.replace('\n', '')
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


def crawl_traverse():
    """
    crawl urls listed in the specified csv file.
    if crawled content is empty, copy the corresponding file from the specified folder.
    """
    
    SENTENCE_LEN = 25
    data_path = "../data/tbrain_train_final_0610.csv"
    df = pd.read_csv(data_path)
    links = df["hyperlink"].to_list()

    PATH_TO_SAVE = '../data/cleaned_crawled_news'
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
                with open(f"../data/news_alltext/{idx}.txt", "r") as f:
                    lines = f.readlines()
                    line = filter_short_lines(lines, SENTENCE_LEN)
                
                with open(f"{PATH_TO_SAVE}/{idx+1}.txt", "w") as f:
                    f.write(line + '\n')

            else:
                with open(f"{PATH_TO_SAVE}/{idx+1}.txt", "w") as f:
                    for paragraph in paragraphs:
                        f.write(paragraph + '\n')

            time.sleep(random.random() * 1.5 + 0.5)

        except:
            time.sleep(random.random() * 1.5 + 0.5)

    return

if __name__ == "__main__":
    # crawl_url()
    # crawl_traverse()
    crawl_mojibake()
