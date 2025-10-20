#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List
import os
import pandas as pd
import numpy as np
from tqdm import trange
from graphgps.run.utils import *
from torch_geometric.graphgym.config import cfg
from torch_geometric.loader import DataLoader
from torch_geometric.graphgym.loss import compute_loss


CWD = os.path.dirname(__file__)


class GraphGPS:
    def __init__(self, save_dir: str, cfg_path: str, 
                 features_generators: List[str] = None, ensemble_size: int = 1, number_of_molecules: int = 1,
                 n_jobs: int = 8, seed: int = 0):
        self.save_dir = save_dir
        if os.path.exists(cfg_path):
            self.cfg_path = cfg_path
        elif os.path.exists(os.path.join(CWD, cfg_path)):
            self.cfg_path = os.path.join(CWD, cfg_path)
        else:
            raise FileNotFoundError(f"Config file {cfg_path} not found.")
        self.features_generators = features_generators
        self.ensemble_size = ensemble_size
        self.number_of_molecules = number_of_molecules
        self.n_jobs = n_jobs
        self.seed = seed
        torch.set_num_threads(self.n_jobs)

    def fit_molalkit(self, train_data, iteration: int = 0):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cfg_init()

        train_data_loader = DataLoader(train_data.dataset_pyg, 
                                       batch_size=cfg.train.batch_size, 
                                       shuffle=True)

        df_loss = pd.DataFrame({})
        self.models = []
        for model_idx in range(self.ensemble_size):
            custom_set_run_dir(cfg, iteration * self.ensemble_size + model_idx)
            seed_everything(self.seed + model_idx)
            model = create_model()
            optimizer = create_optimizer(model.parameters(),
                             new_optimizer_config(cfg))
            scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
            losses = []
            for i in trange(cfg.optim.max_epoch):
                loss = self.train_epoch(train_data_loader, model, optimizer,
                                        scheduler, cfg.optim.batch_accumulation)
                losses.append(loss)
            df_loss[f"model_{model_idx}"] = losses
            self.models.append(model)
        df_loss.to_csv(os.path.join(self.save_dir, "loss-iter%d.csv" % iteration), index=False)
        
    def predict_value(self, pred_data):
        self.cfg_init()
        test_data_loader = DataLoader(pred_data.dataset_pyg,
                                      batch_size=cfg.train.batch_size,
                                      shuffle=False)
        predictions = []
        with torch.no_grad():
            for model_idx, model in enumerate(self.models):
                preds = []
                model.eval()
                for batch in test_data_loader:
                    batch.to(torch.device(cfg.accelerator))
                    pred = model(batch)
                    preds.append(pred[0].detach().cpu().numpy())
                predictions.append(np.concatenate(preds))
        predictions = np.mean(predictions, axis=0).ravel()
        return predictions

    def predict_uncertainty(self, pred_data):
        self.cfg_init()
        if cfg.dataset.task_type == 'regression':
            raise ValueError("Uncertainty estimation is not supported for regression tasks.")
        else:
            preds = self.predict_value(pred_data)
            preds = np.array([preds, 1-preds]).T
            return (0.25 - np.var(preds, axis=1)) * 4

    def cfg_init(self):
        set_cfg(cfg)
        cfg.out_dir = os.path.join(self.save_dir, "graphgps")
        cfg.merge_from_file(self.cfg_path)
        dump_cfg(cfg)
        if self.features_generators is not None:
            cfg.gnn.use_features = True
            n_features = 0
            for fg in self.features_generators:
                if fg in ['rdkit_2d', 'rdkit_2d_normalized']:
                    n_features += 200
                elif fg in ['morgan', 'morgan_count']:
                    n_features += 2048
                else:
                    raise ValueError(f"Unknown features generator: {fg}")
            cfg.gnn.n_features = n_features * self.number_of_molecules
        auto_select_device()

    def train_epoch(self, loader, model, optimizer, scheduler, batch_accumulation):
        model.train()
        optimizer.zero_grad()
        loss_sum = 0.
        for iter, batch in enumerate(loader):
            # batch.split = 'train'
            batch.to(torch.device(cfg.accelerator))
            pred, true = model(batch)
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
            loss.backward()
            # Parameters update after accumulating gradients for given num. batches.
            if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
                if cfg.optim.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   cfg.optim.clip_grad_norm_value)
                optimizer.step()
                optimizer.zero_grad()
            loss_sum += loss.detach().cpu().item()
        scheduler.step()
        return loss_sum / len(loader)
