import ast
import sys
import os

from ckiptagger import (
    data_utils, construct_dictionary, 
    WS, POS, NER
)
from tqdm import tqdm
import pandas as pd
import pickle

with open("./data/NER_tags.pickle", "rb") as f:
    tags = pickle.load(f)

with open("./data/NER_labels.pickle", "rb") as f:
    labels = pickle.load(f)

data_path = "../data/tbrain_train_final_0610.csv"
df = pd.read_csv(data_path)

name_list = df["name"].tolist()
name_list = [set(ast.literal_eval(names)) for names in name_list]

# Calculate F1 Score
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")
import scorer

scorer = scorer.AMRScorer()

score = scorer.calculate_score(labels, name_list)
print(f"F1 Score: {score/5023}")

"""
data_path = "../data/tbrain_train_final_0610.csv"
df = pd.read_csv(data_path)
        
name_list = df["name"].tolist()
name_list = [ast.literal_eval(name) for name in name_list]


DIR_PATH = os.path.abspath(
    f"../data/news"
)
FILES = sorted(os.listdir(DIR_PATH))
FILES.sort(key=len, reverse=False)

FILES = [f"{DIR_PATH}/{path}" for path in FILES]

corpus = []

for p in FILES:
    with open(p, "r") as f:
        text = f.readlines()
        text = " ".join([t.strip("\n") for t in text])
        corpus.append(text)

# CKIP models
ws = WS("../ckip")
ner = NER("../ckip")
pos = POS("../ckip")

NER_LABELS = []
NER_TAGS = []

for doc in tqdm(corpus):
    word_sentence_list = ws([doc])
    pos_sentence_list = pos(word_sentence_list)
    entity_sentence_list = ner(word_sentence_list, pos_sentence_list)

    labels = set()
    tags = []

    for token in entity_sentence_list[0]:
        if token[2] == "PERSON":
            tags.append(token)
            labels.add(token[3])

    NER_LABELS.append(labels)
    NER_TAGS.append(tags)
    # NER_LABELS.append(set([[token[3], token[2]] for token in entity_sentence_list[0] if token[2] == "PERSON"]))

# print(*NER_LABELS, sep="\n----------\n")


with open("./data/NER_tags.pickle", "wb") as f:
    pickle.dump(NER_TAGS, f)

with open("./data/NER_labels.pickle", "wb") as f:
    pickle.dump(NER_LABELS, f)

"""
