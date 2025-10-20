#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from typing import Literal
import os
from functools import partial
import numpy as np
import pandas as pd
import random
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from fast_transformers.masking import LengthMask as LM
from apex import optimizers
from molalkit.models.molformer.tokenizer import MolTranBertTokenizer
from molalkit.models.molformer.data import MolFormerDataset
from molalkit.models.molformer.rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder


CWD = os.path.dirname(__file__)


class LightningModule(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super(LightningModule, self).__init__()
        self.config = config
        self.mode = config.mode
        self.save_hyperparameters(config)
        self.tokenizer=tokenizer
        # Word embeddings layer
        n_vocab, d_emb = len(tokenizer.vocab), config.n_embd
        # input embedding stem
        builder = rotate_builder.from_kwargs(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            query_dimensions=config.n_embd//config.n_head,
            value_dimensions=config.n_embd//config.n_head,
            feed_forward_dimensions=config.n_embd,
            attention_type='linear',
            feature_map=partial(GeneralizedRandomFeatures, n_dims=config.num_feats),
            activation='gelu',
            )
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, config.n_embd)
        self.drop = nn.Dropout(config.d_dropout)
        ## transformer
        self.blocks = builder.get()
        self.lang_model = self.lm_layer(config.n_embd, n_vocab)
        self.train_config = config
        #if we are starting from scratch set seeds
        #########################################
        # protein_emb_dim, smiles_embed_dim, dims=dims, dropout=0.2):
        #########################################

        self.fcs = []
        if config.task_type == 'regression':
            self.loss = torch.nn.L1Loss()
        elif config.task_type == 'binary':
            self.loss = torch.nn.BCEWithLogitsLoss()
        self.net = self.Net(
            config.n_embd, config.num_tasks, dropout=config.dropout,
        )

    class Net(nn.Module):
        def __init__(self, smiles_embed_dim, num_tasks, dropout=0.2):
            super().__init__()
            self.desc_skip_connection = True 
            self.fcs = []  # nn.ModuleList()
            print('dropout is {}'.format(dropout))

            self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.relu1 = nn.GELU()
            self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout2 = nn.Dropout(dropout)
            self.relu2 = nn.GELU()
            self.final = nn.Linear(smiles_embed_dim, num_tasks)

        def forward(self, smiles_emb):
            x_out = self.fc1(smiles_emb)
            x_out = self.dropout1(x_out)
            x_out = self.relu1(x_out)

            if self.desc_skip_connection is True:
                x_out = x_out + smiles_emb

            z = self.fc2(x_out)
            z = self.dropout2(z)
            z = self.relu2(z)
            if self.desc_skip_connection is True:
                z = self.final(z + x_out)
            else:
                z = self.final(z)

            return z

    class lm_layer(nn.Module):
        def __init__(self, n_embd, n_vocab):
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, n_vocab, bias=False)
        def forward(self, tensor):
            tensor = self.embed(tensor)
            tensor = F.gelu(tensor)
            tensor = self.ln_f(tensor)
            tensor = self.head(tensor)
            return tensor

    def on_save_checkpoint(self, checkpoint):
        #save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict['torch_state']=torch.get_rng_state()
        out_dict['cuda_state']=torch.cuda.get_rng_state()
        if np:
            out_dict['numpy_state']=np.random.get_state()
        if random:
            out_dict['python_state']=random.getstate()
        checkpoint['rng'] = out_dict

    def on_load_checkpoint(self, checkpoint):
        #load RNG states each time the model and states are loaded from checkpoint
        rng = checkpoint['rng']
        for key, value in rng.items():
            if key =='torch_state':
                torch.set_rng_state(value)
            elif key =='cuda_state':
                try:
                    torch.cuda.set_rng_state(value)
                except RuntimeError as e:
                    print(f"Warning: Could not set CUDA RNG state: {e}")
            elif key =='numpy_state':
                np.random.set_state(value)
            elif key =='python_state':
                random.setstate(value)
            else:
                print('unrecognized state')

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        if self.pos_emb != None:
            no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.0},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        betas = (0.9, 0.99)
        print('betas are {}'.format(betas))
        learning_rate = self.train_config.lr_start * self.train_config.lr_multiplier
        optimizer = optimizers.FusedLAMB(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(self, batch, train=True):
        idx = batch[0]
        mask = batch[1]
        targets = batch[-1]
        # b, t = idx.size()
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        token_embeddings = x
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        loss_input = sum_embeddings / sum_mask
        pred = self.net.forward(loss_input).squeeze()
        if not train and self.config.task_type == 'binary':
            pred = torch.sigmoid(pred)
        if pred.ndim == 0:
            pred = pred.reshape(-1)
        return pred, targets.float()


class MolFormer:
    def __init__(self, save_dir: str, pretrained_path: str, 
                 task_type: Literal["regression", "binary"] = "regression", num_tasks: int = 1,
                 n_head: int = 12, n_layer: int = 12, n_embd: int = 768, d_dropout: float = 0.1, 
                 dropout: float = 0.1, learning_rate: float = 3e-5, num_feats: int = 32,
                 ensemble_size: int = 1, epochs: int = 50, batch_size: int = 128, 
                 num_workers: int = 8, seed: int = 0,):
        self.save_dir = save_dir
        self.pretrained_path = pretrained_path
        assert os.path.exists(pretrained_path), f"Pretrained model {pretrained_path} not found. Please download it from https://github.com/IBM/molformer"
        self.tokenizer = MolTranBertTokenizer(os.path.join(CWD, 'bert_vocab.txt'))
        hparams = {
            'mode': 'avg',
            'n_head': n_head, 'n_layer': n_layer, 'n_embd': n_embd,
            'd_dropout': d_dropout, 'dropout': dropout, 'num_feats': num_feats,
            'lr_start': learning_rate, 'lr_multiplier': 1, 'task_type': task_type,
            'num_tasks': num_tasks
        }
        self.hparams = argparse.Namespace(**hparams)
        self.task_type = task_type
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit_molalkit(self, train_data, iteration: int = 0):
        """
        Fit the MolFormer model to the training data.

        Parameters
        ----------
        train_data : object
            The training data.
        iteration : int, optional
            The current iteration number, by default 0.
        """
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_data_loader = self.get_dataloader(train_data)
        df_loss = pd.DataFrame({})
        self.models = []
        for model_idx in range(self.ensemble_size):
            pl.seed_everything(self.seed + model_idx)
            model = self.create_model()
            optimizer = model.configure_optimizers()
            losses = []
            for i in trange(self.epochs):
                loss = self.train_epoch(train_data_loader, model, optimizer)
                losses.append(loss)
            df_loss[f"model_{model_idx}"] = losses
            self.models.append(model)
        df_loss.to_csv(os.path.join(self.save_dir, "loss-iter%d.csv" % iteration), index=False)

    def predict_value(self, test_data):
        """
        Predict the properties of the test data using the trained model.

        Parameters
        ----------
        test_data : object
            The test data.

        Returns
        -------
        object
            The predicted properties.
        """
        test_data_loader = self.get_dataloader(test_data, shuffle=False)
        predictions = []
        with torch.no_grad():
            for model_idx, model in enumerate(self.models):
                preds = []
                model.eval()
                for batch in test_data_loader:
                    batch = list(batch)
                    for i in range(3):
                        batch[i] = batch[i].to(self.device)
                    pred, true = model(batch, train=False)
                    preds.append(pred.detach().cpu().numpy())
                predictions.append(np.concatenate(preds))
        predictions = np.mean(predictions, axis=0).ravel()
        return predictions

    def predict_uncertainty(self, pred_data):
        if self.task_type == 'regression':
            # raise ValueError("Uncertainty estimation is not supported for regression tasks.")
            return self.predict_value(pred_data)
        else:
            preds = self.predict_value(pred_data)
            preds = np.array([preds, 1-preds]).T
            return (0.25 - np.var(preds, axis=1)) * 4

    def get_dataloader(self, data, shuffle: bool = True):
        assert data.X_smiles.shape[1] == 1, "Only single-column SMILES data is supported for RNN models."
        assert data.y.shape[1] == 1, "Only single-column target data is supported for RNN models."
        data_ = MolFormerDataset(
            smiles_list=data.X_smiles.ravel().tolist(),
            targets=data.y.ravel().tolist(),
        )

        def collate(batch):
            tokens = self.tokenizer.batch_encode_plus([ smile[0] for smile in batch], padding=True, add_special_tokens=True)
            return (torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask']), torch.tensor([smile[1] for smile in batch]))
        
        return DataLoader(
            data_,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=collate
        )

    def create_model(self):
        return LightningModule.load_from_checkpoint(self.pretrained_path, 
                                                    strict=False, 
                                                    config=self.hparams, 
                                                    tokenizer=self.tokenizer, 
                                                    vocab=len(self.tokenizer.vocab),
                                                    task_type=self.task_type).to(self.device)

    def train_epoch(self, train_data_loader, model, optimizer):
        """
        Train the model for one epoch.

        Parameters
        ----------
        train_data_loader : DataLoader
            The training data loader.
        model : nn.Module
            The model to be trained.
        optimizer : torch.optim.Optimizer
            The optimizer for the model.

        Returns
        -------
        float
            The average loss for the epoch.
        """
        
        model.train()
        epoch_loss = 0
        for batch in train_data_loader:
            batch = list(batch)
            for i in range(3):
                batch[i] = batch[i].to(self.device)
            optimizer.zero_grad()
            pred, true = model(batch)
            loss = model.loss(pred, true)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().to('cpu').item()
        return epoch_loss / len(train_data_loader)
