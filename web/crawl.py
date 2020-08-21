import os
import requests
import unicodedata
import re
from bs4 import BeautifulSoup

def crawl_news(url):
    """
    crawl url pass to the function.
    if crawled content is empty, copy the corresponding file from the specified folder.
    """
    
    header = {'User-Agent': 'Chrome/66.0.3359.181 Safari/537.36'}
    SENTENCE_LEN = 25

    response = requests.get(url, headers=header, timeout=5)
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
    else:
        content = ''

    return content