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

if __name__ == "__main__":
    crawl()
