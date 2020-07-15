# -*- coding: UTF-8 -*-

##############################################
# Import requirements
##############################################
import os
import pickle

from tqdm import tqdm

import torch
import torch.nn
import torch.utils.data
import numpy as np
import pandas as pd
import sklearn

# Local Modules
import dataset
import model
import preprocessor
import tokenizer

##############################################
# Load config and tokenizer
##############################################

experiment_no = 1

load_path = f"models/{experiment_no}"
if not os.path.exists(load_path):
    raise ValueError("Path does not exist, have you been training?")

config_load_path = f"{load_path}/config.pickle"
tokenizer_load_path = f"{load_path}/tokenizer.pickle"

with open(config_load_path, "rb") as f:
    config = pickle.load(f)

with open(tokenizer_load_path, "rb") as f:
    t = pickle.load(f)

##############################################
# Initialize random seed and CUDA
##############################################

device = torch.device("cpu")
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda:0")
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print("Using CPU")

##############################################
# Load test data
##############################################

with open("./data/test.pickle", "rb") as fb:
    test_dataset = pickle.load(fb)

t_df = pd.read_csv("./data/test.csv")
raw_labels = t_df["raw_labels"].tolist()

test_data = dataset.AMLDataset(config=config,
                               dataset=test_dataset,
                               tokenizer=t)

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=config["batch_size"],
                                          collate_fn=test_data.collate_fn,
                                          num_workers=1)

##############################################
# Load model
##############################################

model = model.AMRBaseline(config, t)
model.load_state_dict(torch.load(f"{load_path}/model.pt"))

##############################################
# Validation
##############################################

def validate(model, device, data_loader, tokenizer, load_path, threshold=0.5):

    model.to(device)
    model.eval()

    answers = []

    for index, (documents, label) in enumerate(tqdm(data_loader)):
        documents = documents.to(device)
        label = label.to(device)

        # Get the indexes of the answers higher
        pred_batch = model(documents)

        for pred in pred_batch:
            answers.append([
                (idx, word)
                for idx, word in enumerate(pred)
                if word >= threshold
            ])


    # DOC x (IDX (N), PROB (1))
    return answers

if __name__ == "__main__":

    answers = validate(model, device, test_loader, t, load_path)

    for index, row in enumerate(answers):
        t_row = test_data[index][0]

        tmp_i = 0
        t_label = []
        tmp = []
        for i, char in enumerate(test_data[index][0]):

            if test_data[index][1][i] == 1 and tmp_i == i-1:
                # print("Up")
                tmp.append(char.item())
                tmp_i = i

            elif test_data[index][1][i] == 1 and tmp_i == 0:
                # print("Mid")
                tmp.append(char.item())
                tmp_i = i

            elif test_data[index][1][i] == 1 and tmp_i != i-1:
                # print("Down")
                t_label.append(tmp)
                tmp = []
                tmp.append(char.item())
                tmp_i = i

        if tmp != []:
            t_label.append(tmp)

        pred = []
        tmp = []
        tmp_idx = None
        for idx, prob in row:
            if tmp_idx != idx-1 and tmp_idx is not None:
                pred.append(tmp)
                tmp = []

            tmp.append(t_row[idx].item())
            tmp_idx = idx

        pred.append(tmp)

        print(f"{'-'*50}")
        print(f"{index}|Raw:    \t{raw_labels[index]}")
        print(f"{index}|Label:  \t{list(set([''.join(n) for n in t.decode(t_label) if n != []]))}")
        print(f"{index}|Decoded:\t{list(set([''.join(n) for n in t.decode(pred) if n != []]))}")
