#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Set, Tuple, Union, Dict
import numpy as np
from graphgps.data.data import DatasetFromCSVFile


class DataPoint:
    def __init__(self, smiles: List[str], targets: List[float], id_pyg: int):
        self.smiles = smiles
        self.targets = targets
        self.id_pyg = id_pyg


class Dataset:
    def __init__(self, dataset_pyg_full):
        self.dataset_pyg_full = dataset_pyg_full
        self.data = []
        for i, smiles_list in enumerate(dataset_pyg_full.smiles):
            assert len(smiles_list) == 1, f"Expected 1 SMILES string, got {len(smiles_list)}"
            self.data.append(DataPoint(smiles=smiles_list, targets=dataset_pyg_full.y[i].tolist(), id_pyg=i))

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]

    @property
    def dataset_pyg(self):
        ids_pyg = [data.id_pyg for data in self.data]
        return self.dataset_pyg_full[ids_pyg]

    @property
    def y(self):
        return np.array(self.dataset_pyg.y)


def get_data(path: str, save_dir: str,
             smiles_columns: List[str] = None,
             targets_columns: List[str] = None,
             features_generators: List[str] = None,
             n_jobs: int = 8):
    dataset_pyg_full = DatasetFromCSVFile(data_path=path,
                                          smiles_columns=smiles_columns,
                                          target_columns=targets_columns,
                                          features_generator=features_generators,
                                          root='%s/graphgps' % save_dir)
    return Dataset(dataset_pyg_full)
