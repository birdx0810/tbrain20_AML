# built-in modules
import os
import json
import re
import unicodedata
import time
import random
import sys

# 3rd party modules
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# self-made modules
from data_utils import filter_short_lines

def get_AML_names(file_path=None):
    # get data
    df = pd.read_csv(file_path)

    # get all labels
    labels = df['name'].tolist()
    labels = [json.loads(label.replace("'", '"'))
              for label in labels if label != '[]']

    # get all names
    names = []
    for label in labels:
        names.extend([name for name in label
                      if name not in names])

    # get related names
    related_names = [df[df['name'].apply(lambda names: name in names)].iloc[0]['name']
             for name in names]

    return names, related_names

def search_news_links(name=None):
    # setting query parameter
    search_url = 'https://www.google.com/search'
    header = {'User-Agent': 'Chrome/66.0.3359.181 Safari/537.36'}
    query = {'q': name, 'tbm': 'nws', 'lr': 'lang_zh-TW'}

    # get html
    html = requests.get(search_url, headers=header, params=query)

    # find all the links
    links = BeautifulSoup(html.text, 'html.parser').select(
        'body > div > div > div > div > a')
    links = [link['href'][7:].split('&sa=')[0]
             for link in links
             if 'http' in link['href'] and 'ad' not in link['href']]

    return links

def crawl_news(AML_name, links):
    """
    crawl links pass to the function.
    if crawled content is empty, copy the corresponding file from the specified folder.
    """
    
    header = {'User-Agent': 'Chrome/66.0.3359.181 Safari/537.36'}
    SENTENCE_LEN = 25
    all_news = []

    SAVE_PATH = '../data/cleaned_other_news'
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    for idx, link in enumerate(links[:5]):
        try:
            response = requests.get(link, headers=header, timeout=5)
            response.encoding = "UTF-8"

            soup = BeautifulSoup(response.text, 'html.parser')
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

            if len(paragraphs) != 0:
                content = ''.join(paragraphs)
                content = content.replace('\r', '')
                content = content.replace('\n', '')
                content = content.replace('\t', '')

                all_news.append(content)

                with open(f'{SAVE_PATH}/{AML_name}_{idx}.txt', 'w') as f:
                    f.write('\n'.join(paragraphs))
            else:
                all_news.append('')

        except Exception as e:
            print(f'Link: {link}')
            print(f'Error: {e}')

    return all_news

def main():
    # path
    data_path = "../data/tbrain_train_final_0610.csv"
    output_path = '../data/other_news.csv'

    # get AML names
    AML_names, related_names = get_AML_names(data_path)

    # construct dataframe to store crawled news
    df = pd.DataFrame({'other_news_ID': [], 'person': [], 'hyperlink': [], 'content': [], 'name': []})
    df = df.astype({'other_news_ID': int})

    # crawled news
    index = 0
    name_loader = tqdm(zip(AML_names, related_names),
                       total=len(AML_names),
                       desc='Crawl news')

    for name, related_name in name_loader:
        links = search_news_links(name)
        all_news = crawl_news(name, links)
        for link, news in zip(links, all_news):
            df.loc[index] = [index+1, name, link, news, related_name]
            index += 1

    # output file
    df.to_csv(output_path, index=False)

    return

if __name__ == '__main__':
    main()