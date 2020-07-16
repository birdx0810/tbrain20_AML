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
# Hyperparameters setup
##############################################

experiment_no = 1

config = {}

config["seed"] = 7
config["batch_size"] = 32
config["epochs"] = 200
config["accum_step"] = 1
config["max_seq_len"] = 512

# Optimizer
config["learning_rate"] = 1e-5
config["grad_clip_norm"] = 1.0

# Model Hyperparameters
config["num_rnn_layers"] = 1
config["num_linear_layers"] = 1
config["hidden_dim"] = 300
config["embedding_dim"] = 100
config["bidirectional"] = True
config["dropout"] = 0.1

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
# Tokenizer
##############################################

with open("./data/tokenizer", "rb") as fb:
    t = pickle.load(fb)

##############################################
# Load Training data
##############################################

with open("./data/train.pickle", "rb") as fb:
    train_dataset = pickle.load(fb)

with open("./data/test.pickle", "rb") as fb:
    test_dataset = pickle.load(fb)

train_data = dataset.AMLDataset(config=config,
                                dataset=train_dataset,
                                tokenizer=t)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=config["batch_size"],
                                           collate_fn=train_data.collate_fn,
                                           shuffle=True,
                                           num_workers=1)

##############################################
# Construct model
##############################################

model = model.AMRBaseline(config, t)

criterion = torch.nn.BCELoss()

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]

optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters,
    lr=config["learning_rate"]
)

##############################################
# Train
##############################################

save_path = f"models/{experiment_no}"
if not os.path.exists(save_path):
    os.mkdir(save_path)

config_save_path = f"{save_path}/config.pickle"
tokenizer_save_path = f"{save_path}/tokenizer.pickle"

with open(config_save_path, "wb") as f:
    pickle.dump(config, f)

with open(tokenizer_save_path, "wb") as f:
    pickle.dump(t, f)

def train_model(model, device, data_loader, criterion, optimizer, num_epochs, save_path):

    model_save_path = f"{save_path}/model.pt"
    model.to(device)
    model.train()
    
    best_loss = None

    for epoch in range(num_epochs):
        print(f"epoch: {epoch}")
        total_loss = 0

        for documents, label in tqdm(data_loader):
            documents = documents.to(device)
            label = label.to(device)

            pred = model(documents)

            optimizer.zero_grad()

            loss = criterion(pred, label)
            total_loss += float(loss) / len(train_data)

            # Gradient Accumulation
            # accumulation_loss += loss/config["accum_step"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            optimizer.step()

            # if batch_index % config["accum_step"] == 0:
            #     accumulation_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            #     optimizer.step()

            #     accumulation_loss.detach()
            #     del accumulation_loss
            #     accumulation_loss = 0

        print(f"loss: {total_loss:.10f}")

        if (best_loss is None) or (total_loss < best_loss):
            torch.save(model.state_dict(), model_save_path)
            best_loss = total_loss

    print(f"best loss: {best_loss:.10f}")

train_model(model, device, train_loader, criterion, optimizer, config["epochs"], save_path)
