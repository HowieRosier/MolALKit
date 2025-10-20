#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Tuple, Union
from torch.utils.data import Dataset


class MolFormerDataset(Dataset):
    """Dataset class for molecular data."""
    
    def __init__(self, smiles_list: List[str],
                 targets: List[Union[int, float]]):
        self.smiles_list = smiles_list
        self.targets = targets
        assert len(smiles_list) == len(targets), "Length of SMILES list and targets must be the same."

    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def __getitem__(self, index: int) -> Tuple[List[int], Union[int, float]]:
        return self.smiles_list[index], self.targets[index]
