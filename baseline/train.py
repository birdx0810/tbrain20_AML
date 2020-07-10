# -*- coding: UTF-8 -*-

##############################################
# Import requirements
##############################################

from tqdm import tqdm
import torch
import torch.nn
import torch.utils.data
import numpy as np
import pandas as pd

# from CIDEr.eval import CIDErEvalCap as ciderEval
import os
import pickle

# Local Modules
import tokenizer
import preprocessor
import dataset
import encoder
import decoder

##############################################
# Hyperparameters setup
##############################################

experiment_no = 3

config = {}

config["seed"] = 7
config["batch_size"] = 4
config["epochs"] = 200

# Optimizer
config["learning_rate"] = 1e-4
config["grad_clip_norm"] = 1.0

# Encoder Architecture
config["vgg"] = 11
config["batch_norm"] = True

# Decoder Architecture
config["rnn"] = "GRU"
config["num_rnn_layers"] = 2
config["num_linear_layers"] = 1
config["hidden_dim"] = 300
config["embedding_dim"] = 100
config["dropout"] = 0

##############################################
# Initialize random seed and CUDA
##############################################

device = torch.device('cpu')
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device('cuda:0')
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print("Using CPU")

##############################################
# Preprocess and Load Training data
##############################################

train_data, _ = preprocessor.get_data(seed=config["seed"], num_data=1000)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=config["batch_size"],
                                           collate_fn=train_data.collate_fn,
                                           shuffle=True,
                                           num_workers=1)

##############################################
# Construct model
##############################################

t = tokenizer.Tokenizer()
t.load_vocab("./vocab/glove100_vocab.pickle")
with open("./vocab/glove100d.vec", "rb") as f:
    t.vectors = pickle.load(f)

encoder = encoder.VGGEncoder(config=config, in_features=3)

decoder = decoder.RNNDecoder(config=config, tokenizer=t)

criterion = torch.nn.CrossEntropyLoss()

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in encoder.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {"params": [p for n, p in encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    {
        "params": [p for n, p in decoder.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {"params": [p for n, p in decoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]

# optimizer = torch.optim.AdamW(decoder.parameters(), lr=config["learning_rate"])
optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters,
    lr=config["learning_rate"]
)

##############################################
# Train
##############################################

save_path = f'models/{experiment_no}'
if not os.path.exists(save_path):
    os.mkdir(save_path)

config_save_path = f'{save_path}/config.pickle'
tokenizer_save_path = f'{save_path}/tokenizer.pickle'

with open(config_save_path, 'wb') as f:
    pickle.dump(config, f)

with open(tokenizer_save_path, 'wb') as f:
    pickle.dump(t, f)

def train_model(model, device, data_loader, criterion, optimizer, num_epochs, save_path):

    model_save_path = f"{save_path}/model.pt"

    model.to(device)

    model.train()
    
    best_loss = None

    for epoch in range(num_epochs):
        print(f"epoch: {epoch}")
        total_loss = 0

        for document, names in tqdm(data_loader):
            document = document.to(device)
            # captions = captions.to(device)

            input_captions = captions[:, :-1].to(device)
            output_captions = captions[:, 1:].to(device)

            features = model(images)

            # B x V x S
            outputs = outputs.transpose(-1,-2)

            loss = criterion(outputs, output_captions)
            total_loss += float(loss) / len(train_data)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), config["grad_clip_norm"])
            optimizer.step()

        print(f'loss: {total_loss:.10f}')

        if (best_loss is None) or (total_loss < best_loss):
            torch.save(encoder.state_dict(), encoder_save_path)
            torch.save(decoder.state_dict(), decoder_save_path)
            best_loss = total_loss

    print(f'best loss: {best_loss:.10f}')

train_model(model, device, train_loader, criterion, optimizer, config["epochs"], save_path)

