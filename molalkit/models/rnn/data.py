#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Tuple, Union
from torch.utils.data import Dataset
from molalkit.models.rnn.tokenizer import BaseTokenizer


class MoleculeDataset(Dataset):
    """Dataset class for molecular data."""
    
    def __init__(self, smiles_list: List[str],
                 targets: List[Union[int, float]],
                 tokenizer: BaseTokenizer):
        self.smiles_list = smiles_list
        self.targets = targets
        self.tokenizer = tokenizer
        self.data = [tokenizer.encode(smiles, 
                                      generative=False, 
                                      padding=False) for smiles in smiles_list]

    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def __getitem__(self, index: int) -> Tuple[List[int], Union[int, float]]:
        return self.data[index], self.targets[index]
