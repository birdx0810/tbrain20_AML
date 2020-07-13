# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn

class AMRBaseline(torch.nn.Module):
    def __init__(self, config, tokenizer):
        super(AMRBaseline, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocabulary_size()
        self.embedding_weight = self.tokenizer.vectors

        self.pad_token_idx = tokenizer.pad_token_idx
        self.unk_token_idx = tokenizer.unk_token_idx
        self.bos_token_idx = tokenizer.bos_token_idx
        self.eos_token_idx = tokenizer.eos_token_idx

        self.num_rnn_layers = config['num_rnn_layers']

        # Embedding layer
        self.embedding_layer = torch.nn.Embedding(num_embeddings=self.vocab_size,
                                                  embedding_dim=config["embedding_dim"],
                                                  padding_idx=self.pad_token_idx)

        # if self.embedding_weight != []:
        #     self.embedding_layer.weight = torch.nn.Parameter(torch.Tensor(self.embedding_weight))

        with torch.no_grad():
            for parameter in self.embedding_layer.parameters():
                parameter.normal_(mean=0.0, std=0.1)

        # RNN layer
        self.rnn_layer = torch.nn.LSTM(input_size=config["embedding_dim"],
                                       hidden_size=config["hidden_dim"],
                                       num_layers=config["num_rnn_layers"],
                                       bidirectional=config["bidirectional"],
                                       dropout=config["dropout"],
                                       batch_first=True)

        with torch.no_grad():
            for parameter in self.rnn_layer.parameters():
                parameter.normal_(mean=0.0, std=0.1)

        # Linear Layer
        linear = []

        for _ in range(config["num_linear_layers"]):
            if config["bidirectional"] and _ == 0:
                linear.append(torch.nn.Linear(config["hidden_dim"]*2, config["hidden_dim"]))
            else:
                linear.append(torch.nn.Linear(config["hidden_dim"], config["hidden_dim"]))

            linear.append(torch.nn.ReLU())
            linear.append(torch.nn.Dropout(config["dropout"]))

        linear.append(torch.nn.Linear(
            config["hidden_dim"],
            1
        ))

        linear.append(torch.nn.Sigmoid())

        self.sequential = torch.nn.Sequential(*linear)

        with torch.no_grad():
            for parameter in self.sequential.parameters():
                parameter.normal_(mean=0.0, std=0.1)

    def forward(self, x):

        # B x S x E
        x = self.embedding_layer(x)

        # B x S x H
        ht, _ = self.rnn_layer(x)

        # B x S
        yt = self.sequential(ht).squeeze()

        return yt