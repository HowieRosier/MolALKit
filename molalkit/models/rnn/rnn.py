#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Literal
from tqdm import trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from mgktools.data.data import Dataset
from molalkit.models.rnn.data import MoleculeDataset
from molalkit.models.rnn.tokenizer import BaseTokenizer


class MoleculeRNN(nn.Module):
    def __init__(self,
                 task_type: Literal["binary", "regression"],
                 vocab_size: int,
                 rnn_type: Literal["LSTM", "GRU"] = "GRU",
                 embedding_dim: int = 64,
                 depth: int = 2,
                 hidden_size: int = 128,
                 dropout: float = 0.0,
                 ffn_num_layers: int = 3):
        super(MoleculeRNN, self).__init__()
        self.task_type = task_type
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim)
        self.rnn_type = rnn_type
        self.depth = depth
        self.hidden_size = hidden_size

        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_size,
                depth,
                batch_first=True,
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_size,
                depth,
                batch_first=True,
            )
        else:
            raise ValueError("rnn_type must be either LSTM or GRU")
        
        dropout = nn.Dropout(dropout)
        activation = nn.ReLU()
        ffn = []
        for i in range(ffn_num_layers):
            if i == ffn_num_layers - 1:
                ffn += [
                    dropout,
                    nn.Linear(hidden_size, 1)
                ]
            else:
                ffn += [
                    dropout,
                    nn.Linear(hidden_size, hidden_size),
                    activation
                ]
        self.ffn = nn.Sequential(*ffn)

        if self.task_type == "binary":
            self.sigmoid = nn.Sigmoid()
        # Initialize weights
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, x):
        batch_size = x.shape[0]
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, sequence_length, embedding_dim)
        if self.rnn_type == "LSTM":
            h0 = torch.zeros(self.depth, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.depth, batch_size, self.hidden_size).to(x.device)
            output, _ = self.rnn(embedded, (h0, c0))
        elif self.rnn_type == "GRU":  # GRU
            h0 = torch.zeros(self.depth, batch_size, self.hidden_size).to(x.device)
            output, _ = self.rnn(embedded, h0)
        
        output = output[: ,-1]
        # return self.ffn(output)
        if self.task_type == "binary":
            return self.sigmoid(self.ffn(output))
        else:
            return self.ffn(output)


class RNN:
    def __init__(self, tokenizer: BaseTokenizer,
                 task_type: Literal["binary", "regression"], rnn_type: Literal["LSTM", "GRU"] = "LSTM",
                 embedding_size: int = 128, hidden_size: int = 256, depth: int = 2, dropout: float = 0.1,
                 ffn_num_layers: int = 3, epochs: int = 30, batch_size: int = 64):
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.rnn_type = rnn_type
        self.embedding_dim = embedding_size
        self.depth = depth
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.ffn_num_layers = ffn_num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def fit_molalkit(self, train_data: Dataset, iteration: int = 0):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # initialize model
        model = MoleculeRNN(
            task_type=self.task_type,
            vocab_size=len(self.tokenizer.vocab),
            rnn_type=self.rnn_type,
            embedding_dim=self.embedding_dim,
            depth=self.depth,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            ffn_num_layers=self.ffn_num_layers
        )
        model = model.to(self.device)

        # initialize data loader
        train_data_loader = self.get_dataloader(train_data)

        if self.task_type == "binary":
            loss = nn.BCELoss()
        else:
            loss = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         T_0=5,
                                                                         T_mult=2)
        for epoch in trange(self.epochs):
            model.train()
            epoch_loss = 0.
            for batch in train_data_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                y_pred = model(x)
                batch_loss = loss(y_pred, y.unsqueeze(1))
                batch_loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += batch_loss.item()
            # print(epoch_loss / len(train_data_loader))
        self.model = model

    def predict_value(self, pred_data):
        pred_data_loader = self.get_dataloader(pred_data)
        with torch.no_grad():
            self.model.eval()
            y_pred = []
            for batch in pred_data_loader:
                x, _ = batch
                x = x.to(self.device)
                y_pred.append(self.model(x).cpu().numpy())
            y_pred = np.concatenate(y_pred)
            return y_pred.ravel()

    def predict_uncertainty(self, pred_data):
        assert self.task_type == "binary", "Uncertainty estimation is only available for binary classification"
        preds = self.predict_value(pred_data)
        preds = np.array([preds, 1-preds]).T
        return (0.25 - np.var(preds, axis=1)) * 4

    def get_dataloader(self, data: Dataset):
        assert data.X_smiles.shape[1] == 1, "Only single-column SMILES data is supported for RNN models."
        assert data.y.shape[1] == 1, "Only single-column target data is supported for RNN models."
        data_ = MoleculeDataset(
            smiles_list=data.X_smiles.ravel().tolist(),
            targets=data.y.ravel().tolist(),
            tokenizer=self.tokenizer
        )

        def pad_collate(batch):
            """
            Put the sequences of different lengths in a minibatch by paddding.
            """
            # embedding layer takes long tensors
            x = [torch.tensor(x[0], dtype=torch.long) for x in batch]
            y = [torch.tensor(x[1], dtype=torch.float) for x in batch]

            x_padded = pad_sequence(
                x, 
                batch_first=True,
                padding_value=self.tokenizer.vocab["<pad>"]
            )

            return x_padded, torch.stack(y)
        
        return DataLoader(
            data_, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=pad_collate
        )
