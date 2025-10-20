#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Optional
import re
import selfies as sf


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers.

    Methods
    -------
    tokenize(smiles: str) -> List[str]
        Tokenizes a SMILES string.
    create_vocabulary(smiles_list: List[str]) -> Dict[str, int]
        Creates a vocabulary from a list of SMILES strings.
    encode(smiles: str) -> List[int]
        Encodes a SMILES string into a list of integers.
    """

    @abstractmethod
    def tokenize(self, smiles: str) -> List[str]:
        pass

    @abstractmethod
    def create_vocabulary(self, smiles_list: List[str]) -> Dict[str, int]:
        pass
    
    @abstractmethod
    def encode(self, smiles: str) -> List[int]:
        pass


class SMILESTokenizer(BaseTokenizer):
    """
    A simple regex-based tokenizer adapted from the deepchem smiles_tokenizer package.
    SMILES regex pattern for the tokenization is designed by Schwaller et. al., ACS Cent. Sci 5 (2019).

    Attributes
    ----------
    regex_pattern : str
        The regex pattern used for tokenizing SMILES strings.
    regex : re.Pattern
        The compiled regex pattern.
    special_tokens : List[str]
        List of special tokens used in tokenization.

    Methods
    -------
    tokenize(smiles: str) -> List[str]
        Tokenizes a SMILES string.
    create_vocabulary(smiles_list: List[str]) -> Dict[str, int]
        Creates a vocabulary from a list of SMILES strings.
    encode(smiles: str) -> List[int]
        Encodes a SMILES string into a list of integers.
    """

    def __init__(self):
        """
        Initializes the SMILESTokenizer with a regex pattern and special tokens.
        """
        self.regex_pattern = (
            r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\."
            r"|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
        )
        self.regex = re.compile(self.regex_pattern)
        self.special_tokens = ['<eos>', '<sos>', '<pad>']

    def tokenize(self, smiles):
        """
        Tokenizes SMILES string.

        Parameters
        ----------
        smiles : str
            Input SMILES string.

        Returns
        -------
        List[str]
            A list of tokens.
        """
        tokens = [token for token in self.regex.findall(smiles)]
        return tokens

    def create_vocabulary(self, smiles_list: List[str]) -> Dict[str, int]:
        """
        Create a vocabulary dictionary from a list of SMILES strings.

        Parameters:
        ----------
            smiles_list (List[str]): List of SMILES strings to process.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping tokens to indices.
        """
        assert not hasattr(self, "vocab"), "Vocabulary already exists. Create a new tokenizer instead."
        # Collect all unique tokens
        tokens = set()
        max_length = 0
        for smiles in smiles_list:
            tokens_ = self.tokenize(smiles)
            tokens.update(tokens_)
            if len(tokens) > max_length:
                max_length = len(tokens)
        
        # Create vocabulary with indices
        tokens = list(sorted(tokens)) + list(self.special_tokens)
        vocab_dict = {token: idx for idx, token in enumerate(tokens)}

        self.vocab = vocab_dict
        self.max_length = max_length
        return vocab_dict

    def encode(self, smiles: str, generative: bool = False, padding: bool = True) -> List[int]:
        """
        Encode a SMILES string as a list of token indices.

        Parameters
        ----------
            smiles (str): Input SMILES string.

        Returns
        -------
        List[int]
            List of token indices.
        """
        assert hasattr(self, "vocab"), "Vocabulary not created. Call create_vocabulary() first."
        tokens = self.tokenize(smiles)
        ints = [self.vocab[token] for token in tokens]
        if generative:
            ints = [self.vocab['<sos>'], *ints, self.vocab['<eos>']]
            max_length = self.max_length + 2
        else:
            max_length = self.max_length
        if padding:
            ints += [self.vocab['<pad>']] * (max_length - len(ints))

        return ints


class SELFIESTokenizer(BaseTokenizer):
    """
    A tokenizer for converting SMILES to SELFIES and generating vocabulary.
    Uses the SELFIES library for molecular string representation.
    """

    def __init__(self):
        """
        Initializes the SELFIESTokenizer with special tokens.
        """
        self.special_tokens = ['.', '<eos>', '<sos>', '<pad>']

    def tokenize(self, smiles: str) -> List[str]:
        """
        Tokenizes SMILES string.

        Parameters
        ----------
        smiles : str
            Input SMILES string.

        Returns
        -------
        List[str]
            A list of tokens.
        """
        selfies = sf.encoder(smiles)
        return list(sf.split_selfies(selfies))

    def create_vocabulary(self, smiles_list: List[str]) -> Dict[str, int]:
        """
        Create a vocabulary dictionary from a list of SMILES strings.

        Parameters:
        ----------
            smiles_list (List[str]): List of SMILES strings to process.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping tokens to indices.
        """
        assert not hasattr(self, "vocab"), "Vocabulary already exists. Create a new tokenizer instead."
        # Collect all unique tokens
        tokens = set()
        max_length = 0
        for smiles in smiles_list:
            tokens_ = self.tokenize(smiles)
            tokens.update(tokens_)
            if len(tokens) > max_length:
                max_length = len(tokens)
        
        # Create vocabulary with indices
        tokens = list(sorted(tokens)) + list(self.special_tokens)
        vocab_dict = {token: idx for idx, token in enumerate(tokens)}

        self.vocab = vocab_dict
        self.max_length = max_length
        return vocab_dict

    def encode(self, smiles: str, generative: bool = False, padding: bool = True) -> List[int]:
        """
        Encode a SMILES string as a list of token indices.

        Parameters
        ----------
            smiles (str): Input SMILES string.

        Returns
        -------
        List[int]
            List of token indices.
        """

        assert hasattr(self, "vocab"), "Vocabulary not created. Call create_vocabulary() first."
        tokens = self.tokenize(smiles)
        ints = [self.vocab[token] for token in tokens]
        if generative:
            ints = [self.vocab['<sos>'], *ints, self.vocab['<eos>']]
            max_length = self.max_length + 2
        else:
            max_length = self.max_length
        if padding:
            ints += [self.vocab['<pad>']] * (max_length - len(ints))

        return ints
