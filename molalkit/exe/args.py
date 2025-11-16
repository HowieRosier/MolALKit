#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Literal, Tuple, Callable
from tap import Tap
import os
import shutil
import json
import math
import copy
import pandas as pd
import numpy as np
from mgktools.data.split import data_split_index
from mgktools.evaluators.metric import Metric
from mgktools.features_mol.features_generators import FeaturesGenerator
from molalkit.active_learning.selector import (
    RandomSelector, ClusterRandomSelector, 
    ExplorativeSelector, ClusterExplorativeSelector,
    PartialQueryExplorativeSelector, PartialQueryClusterExplorativeSelector,
    ExploitiveSelector, ClusterExploitiveSelector,
    PartialQueryExploitiveSelector, PartialQueryClusterExploitiveSelector,
)
from molalkit.active_learning.forgetter import (
    FirstForgetter, RandomForgetter, 
    MinOOBUncertaintyForgetter, MaxOOBUncertaintyForgetter,
    MinOOBErrorForgetter, MaxOOBErrorForgetter, 
    MinLOOUncertaintyForgetter, MaxLOOUncertaintyForgetter,
    MinLOOErrorForgetter, MaxLOOErrorForgetter
)
from molalkit.active_learning.utils import get_topn_idx
from molalkit.data.datasets import DATA_DIR
from molalkit.data.utils import get_data, get_subset_from_uidx
from molalkit.exe.logging import create_logger
from molalkit.exe.utils import (
    read_df,
    get_features_generators_from_config,
    get_kernel_from_config,
    get_model_from_config,
    add_error_rate_to_labels,
    apply_error_rate_to_id2datapoint,
)
from molalkit.models.base import BaseSklearnModel
from molalkit.models.configs import MODEL_DIR


class CommonArgs(Tap):
    save_dir: str
    """the output directory."""
    n_jobs: int = 1
    """the cpu numbers used for parallel computing."""
    logger_name: str = "alb_output"
    """the prefix of the output logger file: verbose.log and quite.log"""
    verbose: int = 1
    """the level of verbosity. 0, 1, 2"""
    seed: int = 0
    """random seed."""

    def process_args(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = create_logger(self.logger_name, save_dir=self.save_dir, verbose=self.verbose)
        np.random.seed(self.seed)


class DatasetArgs(CommonArgs):
    data_public = None
    """Use public data sets."""
    data_path: str = None
    """the Path of input data CSV file."""
    data_path_training: str = None
    """the Path of input data CSV file for training set."""
    data_path_pool: str = None
    """the Path of input data CSV file for pool set."""
    data_path_val: str = None
    """the Path of input data CSV file for validation set."""
    smiles_columns: List[str] = None
    """
    For pure compounds.
    Name of the columns containing single SMILES or InChI string.
    """
    targets_columns: List[str] = None
    """
    Name of the columns containing target values.
    """
    features_columns: List[str] = None
    """
    Name of the columns containing additional features_mol such as temperature, 
    pressuer.
    """
    task_type: Literal["regression", "binary", "multiclass"] = None
    """
    Type of task.
    """
    split_type: Literal["random", "scaffold_random", "scaffold_order"] = None
    """Method of splitting the data into active learning/validation."""
    split_sizes: List[float] = None
    """Split proportions for active learning/validation sets."""
    full_val: bool = False
    """validate the performance of active learning on the full dataset."""
    init_size: int = 2
    """number of samples as the initial."""
    error_rate: float = None
    """the percent of the training set that will be affected by error (0-1), valid only for binary classification."""

    def get_train_pool_split_index(self, df_al: pd.DataFrame) -> Tuple[List[int], List[int]]:
        if self.init_size >= len(df_al):
            train_index, pool_index = list(range(len(df_al))), []
        else:
            if self.task_type == "regression":
                train_index, pool_index = data_split_index(
                    n_samples=len(df_al),
                    mols=df_al[self.smiles_columns[0]] if self.smiles_columns is not None else None,
                    split_type="random",
                    sizes=[self.init_size / len(df_al), 1 - self.init_size / len(df_al)],
                    seed=self.seed)
            else:
                train_index, pool_index = data_split_index(
                    n_samples=len(df_al),
                    mols=df_al[self.smiles_columns[0]] if self.smiles_columns is not None else None,
                    targets=df_al[self.targets_columns[0]],
                    split_type="init_al",
                    n_samples_per_class=1,
                    seed=self.seed)
                # randomly select self.init_size - 2 samples from the pool set to be the training set
                if self.init_size > 2:
                    train_index.extend(np.random.choice(pool_index, self.init_size - 2, replace=False))
                    pool_index = [i for i in pool_index if i not in train_index]
        return train_index, pool_index

    def process_args(self) -> None:
        super().process_args()
        if self.data_public == "freesolv" or self.data_public == "test_regression":
            self.data_path = os.path.join(DATA_DIR, "%s.csv" % self.data_public)
            self.smiles_columns = ["smiles"]
            self.targets_columns = ["freesolv"]
            self.task_type = "regression"
        elif self.data_public == "delaney":
            self.data_path = os.path.join(DATA_DIR, "%s.csv" % self.data_public)
            self.smiles_columns = ["smiles"]
            self.targets_columns = ["logSolubility"]
            self.task_type = "regression"
        elif self.data_public == "lipo":
            self.data_path = os.path.join(DATA_DIR, "%s.csv" % self.data_public)
            self.smiles_columns = ["smiles"]
            self.targets_columns = ["lipo"]
            self.task_type = "regression"
        elif self.data_public == "pdbbind_refined":
            self.data_path = os.path.join(DATA_DIR, "%s.csv" % self.data_public)
            self.smiles_columns = ["smiles"]
            self.targets_columns = ["-logKd/Ki"]
            self.task_type = "regression"
        elif self.data_public == "pdbbind_full":
            self.data_path = os.path.join(DATA_DIR, "%s.csv" % self.data_public)
            self.smiles_columns = ["smiles"]
            self.targets_columns = ["-logKd/Ki"]
            self.task_type = "regression"
        elif self.data_public in ["ld50_zhu", "caco2_wang", "solubility_aqsoldb", "ppbr_az", "vdss_lombardo",
                                  "Half_Life_Obach", "Clearance_Hepatocyte_AZ"]:
            self.data_path = os.path.join(DATA_DIR, "%s.csv" % self.data_public)
            self.smiles_columns = ["Drug"]
            self.targets_columns = ["Y"]
            self.task_type = "regression"
        elif self.data_public == "bbbp":
            self.data_path = os.path.join(DATA_DIR, "%s.csv" % self.data_public)
            self.smiles_columns = ["smiles"]
            self.targets_columns = ["p_np"]
            self.task_type = "binary"
        elif self.data_public == "bace":
            self.data_path = os.path.join(DATA_DIR, "%s.csv" % self.data_public)
            self.smiles_columns = ["mol"]
            self.targets_columns = ["Class"]
            self.task_type = "binary"
        elif self.data_public == "hiv":
            self.data_path = os.path.join(DATA_DIR, "%s.csv" % self.data_public)
            self.smiles_columns = ["smiles"]
            self.targets_columns = ["HIV_active"]
            self.task_type = "binary"
        elif self.data_public in ["ames", "carcinogens_lagunin", "dili", "herg", "skin", "hia_hou", "pgp_broccatelli",
                                  "bioavailability_ma", "clintox", "bbb_martins", "CYP1A2_Veith",
                                  "CYP2C9_Substrate_CarbonMangels", "CYP2C9_Veith", "CYP2C19_Veith",
                                  "CYP2D6_Substrate_CarbonMangels", "CYP2D6_Veith", "CYP3A4_Veith",
                                  "CYP3A4_Substrate_CarbonMangels"]:
            self.data_path = os.path.join(DATA_DIR, "%s.csv" % self.data_public)
            self.smiles_columns = ["Drug"]
            self.targets_columns = ["Y"]
            self.task_type = "binary"
        elif self.data_public == "human_liver_microsome_stability":
            self.data_path = os.path.join(DATA_DIR, "%s.csv" % self.data_public)
            self.smiles_columns = ["SMILES"]
            self.targets_columns = ["LOG HLM_CLint (mL/min/kg)"]
            self.task_type = "regression"
        elif self.data_public == "rat_liver_microsome_stability":
            self.data_path = os.path.join(DATA_DIR, "%s.csv" % self.data_public)
            self.smiles_columns = ["SMILES"]
            self.targets_columns = ["LOG RLM_CLint (mL/min/kg)"]
            self.task_type = "regression"
        elif self.data_public == "MDRR1-MDCK_efflux_ratio":
            self.data_path = os.path.join(DATA_DIR, "%s.csv" % self.data_public)
            self.smiles_columns = ["SMILES"]
            self.targets_columns = ["LOG MDR1-MDCK ER (B-A/A-B)"]
            self.task_type = "regression"
        elif self.data_public == "aqueous_solubility":
            self.data_path = os.path.join(DATA_DIR, "%s.csv" % self.data_public)
            self.smiles_columns = ["SMILES"]
            self.targets_columns = ["LOG SOLUBILITY PH 6.8 (ug/mL)"]
            self.task_type = "regression"
        elif self.data_public == "human_plasma_protein_binding":
            self.data_path = os.path.join(DATA_DIR, "%s.csv" % self.data_public)
            self.smiles_columns = ["SMILES"]
            self.targets_columns = ["LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)"]
        elif self.data_public == "rat_plasma_protein_binding":
            self.data_path = os.path.join(DATA_DIR, "%s.csv" % self.data_public)
            self.smiles_columns = ["SMILES"]
            self.targets_columns = ["LOG PLASMA PROTEIN BINDING (RAT) (% unbound)"]

        if self.split_type is not None and self.split_type.startswith("scaffold"):
            assert len(self.smiles_columns) == 1, f"Scaffold split is valid only for single SMILES column, but {len(self.smiles_columns)} are provided."

        assert len(self.targets_columns) == 1, "multi-task active learning is not implemented yet."

        if self.data_path is not None:
            # All data comes from the same file.
            assert self.data_path_val is None and self.data_path_training is None and self.data_path_pool is None
            df = read_df(self.data_path, self.task_type, self.targets_columns)
            if "uidx" not in df:
                df["uidx"] = range(len(df))
            if self.full_val:
                # use the full dataset as validation set
                assert self.split_type is None
                assert self.split_sizes is None
                assert self.error_rate is None
                df.to_csv("%s/val.csv" % self.save_dir, index=False)
                df_al = df
            else:
                # split the dataset into active learning and validation sets
                al_index, val_index = data_split_index(
                    n_samples=len(df),
                    mols=df[self.smiles_columns[0]] if self.smiles_columns is not None else None,
                    # targets=df[self.targets_columns[0]],
                    split_type=self.split_type,
                    sizes=self.split_sizes,
                    seed=self.seed,
                    logger=self.logger)
                df[df.index.isin(val_index)].to_csv("%s/val.csv" % self.save_dir, index=False)
                df_al = df[df.index.isin(al_index)].copy()
                if self.error_rate is not None:
                    assert self.task_type == "binary", "error_rate is valid only for binary classification."
                    # randomly select a portion of the training set to be affected by error
                    add_error_rate_to_labels(df_al, self.error_rate, self.targets_columns[0])
            # split the active learning set into training and pool sets
            train_index, pool_index = self.get_train_pool_split_index(df_al)
            df_al.iloc[train_index].to_csv("%s/train_init.csv" % self.save_dir, index=False)
            df_al.iloc[pool_index].to_csv("%s/pool_init.csv" % self.save_dir, index=False)
        else:
            # data comes from 3 different files.
            assert self.data_path_training is not None, "please provide input data"
            if self.data_path_pool is None:
                df_al = read_df(self.data_path_training, self.task_type, self.targets_columns)
                train_index, pool_index = self.get_train_pool_split_index(df_al)
                df_al.iloc[train_index].to_csv("%s/train_init.csv" % self.save_dir, index=False)
                df_al.iloc[pool_index].to_csv("%s/pool_init.csv" % self.save_dir, index=False)
            else:
                df_train = read_df(self.data_path_training, self.task_type, self.targets_columns)
                df_train.to_csv("%s/train_init.csv" % self.save_dir, index=False)
                df_pool = read_df(self.data_path_pool, self.task_type, self.targets_columns)
                df_pool.to_csv("%s/pool_init.csv" % self.save_dir, index=False)
            if self.data_path_val is None:
                df_train.sample(0).to_csv("%s/val.csv" % self.save_dir, index=False)
                df = pd.concat([pd.read_csv(f) for f in ["%s/train_init.csv" % self.save_dir,
                                                         "%s/pool_init.csv" % self.save_dir]])
            else:
                df_val = read_df(self.data_path_val, self.task_type, self.targets_columns)
                df_val.to_csv("%s/val.csv" % self.save_dir, index=False)
                df = pd.concat([pd.read_csv(f) for f in ["%s/train_init.csv" % self.save_dir,
                                                         "%s/pool_init.csv" % self.save_dir,
                                                         "%s/val.csv" % self.save_dir]])
            if "uidx" not in df:
                df["uidx"] = range(len(df))
                df_train = pd.read_csv("%s/train_init.csv" % self.save_dir)
                df_train["uidx"] = range(len(df_train))
                df_pool = pd.read_csv("%s/pool_init.csv" % self.save_dir)
                df_pool["uidx"] = range(len(df_pool))
                df_pool["uidx"] += len(df_train)
                df_val = pd.read_csv("%s/val.csv" % self.save_dir)
                df_val["uidx"] = range(len(df_val))
                df_val["uidx"] += len(df_train) + len(df_pool)
                if self.error_rate is not None:
                    add_error_rate_to_labels(df_train, self.error_rate, self.targets_columns[0])
                    add_error_rate_to_labels(df_pool, self.error_rate, self.targets_columns[0])
                df_train.to_csv("%s/train_init.csv" % self.save_dir, index=False)
                df_pool.to_csv("%s/pool_init.csv" % self.save_dir, index=False)
                df_val.to_csv("%s/val.csv" % self.save_dir, index=False)
        df.to_csv("%s/full.csv" % self.save_dir, index=False)


class ModelArgs(Tap):
    model_configs: List[str]
    """ A list of config files contain all information of the machine learning model for performance evaluation.
        The first one will be used as the selector model, and the rest will be used as evaluators."""
    
    # L2 (weight decay)
    weight_decay: float = 0.0
    """L2 penalty (weight decay) for optimizer."""

    perturb_sigma: float = 0.0
    """Standard deviation of Gaussian noise for perturbation (N(0, sigma^2))."""

    # CBP (Continual Backpropagation) parameters
    cbp: bool = False
    """Enable Continual Backpropagation for neural plasticity"""
    maturity_threshold: int = 100
    """Number of training steps before a neuron is considered mature enough for replacement"""
    replacement_rate: float = 0.001
    """Fraction of neurons to replace at each replacement step"""
    decay_rate: float = 0.95
    """Decay rate for neuron utility tracking"""
    util_type: str = 'contribution'
    """Utility calculation method: 'contribution', 'weight', 'adaptation', or 'random'"""

    @property
    def model_configs_dict(self) -> List[Dict]:
        return [json.loads(open(self.find_model_config_file(m)).read()) for m in self.model_configs]

    @staticmethod
    def find_model_config_file(config_name):
        if os.path.exists(config_name):
            # try to find the config file in the current directory
            return config_name
        else:
            # try to find the config file in MolALKit"s model directory
            stored_config_name = f"{MODEL_DIR}/{config_name}"
            assert os.path.exists(stored_config_name), f"{config_name} not found."
            return stored_config_name


class DatasetModelArgs(DatasetArgs, ModelArgs):
    @property
    def features_generators(self) -> Optional[List[List[FeaturesGenerator]]]:
        return [get_features_generators_from_config(model_config) for model_config in self.model_configs_dict]

    @property
    def kernels(self) -> List[Callable]:
        if not hasattr(self, "_kernels"):
            self._kernels = [get_kernel_from_config(
                model_config=model_config,
                dataset=self.datasets_full[i],
                kernel_pkl_path="%s/kernels_%d.pkl" % (self.save_dir, i),
            ) for i, model_config in enumerate(self.model_configs_dict)]
        return self._kernels

    @property
    def models(self) -> List[BaseSklearnModel]:
        if not hasattr(self, "_models"):
            self._models = [get_model_from_config(
                model_config=model_config,
                dataset=self.datasets_full[i],
                task_type=self.task_type,
                save_dir=self.save_dir,
                data_path="%s/full.csv" % self.save_dir,
                smiles_columns=self.smiles_columns,
                targets_columns=self.targets_columns,
                features_generators=self.features_generators[i],
                kernel=self.kernels[i],
                n_jobs=self.n_jobs,
                seed=self.seed,
                logger=self.logger,
                # L2 weight decay
                weight_decay=self.weight_decay,
                perturb_sigma=self.perturb_sigma,
                # Pass CBP parameters - prioritize config file values, then command line args
                cbp=model_config.get('cbp', self.cbp),
                maturity_threshold=model_config.get('maturity_threshold', self.maturity_threshold),
                replacement_rate=model_config.get('replacement_rate', self.replacement_rate),
                decay_rate=model_config.get('decay_rate', self.decay_rate),
                util_type=model_config.get('util_type', self.util_type)
            ) for i, model_config in enumerate(self.model_configs_dict)]
        return self._models

    @property
    def datasets_full(self) -> List:
        if not hasattr(self, "_datasets_full"):
            self._datasets_full = [get_data(
                data_format=model_config["data_format"],
                save_dir=self.save_dir,
                file_name="full.csv",
                smiles_columns=self.smiles_columns,
                targets_columns=self.targets_columns,
                features_columns=self.features_columns,
                features_generators=self.features_generators[i],
                features_combination=model_config.get("features_combination"),
                graph_kernel_type=model_config.get("graph_kernel_type") or "no",
                n_jobs=self.n_jobs) for i, model_config in enumerate(self.model_configs_dict)]
        return self._datasets_full

    @property
    def datasets_empty(self) -> List:
        if not hasattr(self, "_datasets_empty"):
            self._datasets_empty = []
            for dataset in self.datasets_full:
                dataset_empty = copy.copy(dataset)
                dataset_empty.data = []
                self._datasets_empty.append(dataset_empty)
        return self._datasets_empty.copy()

    @property
    def id2datapoints(self) -> List[Dict]:
        if not hasattr(self, "_id2datapoints"):
            self._id2datapoints = [{data.uidx: data for data in dataset} for dataset in self.datasets_full]
            if self.error_rate:
                for id2datapoint in self._id2datapoints:
                    df = pd.read_csv("%s/train_init.csv" % self.save_dir)
                    apply_error_rate_to_id2datapoint(id2datapoint, df, self.targets_columns[0])
                    df = pd.read_csv("%s/pool_init.csv" % self.save_dir)
                    apply_error_rate_to_id2datapoint(id2datapoint, df, self.targets_columns[0])
        return self._id2datapoints

    @property
    def datasets_train(self):
        if not hasattr(self, "_datasets_train"):
            df = pd.read_csv("%s/train_init.csv" % self.save_dir)
            self._datasets_train = [get_subset_from_uidx(dataset, id2datapoint, df["uidx"].tolist()) 
                                    for dataset, id2datapoint in zip(self.datasets_empty, self.id2datapoints)]
        return self._datasets_train
    
    @property
    def datasets_pool(self):
        if not hasattr(self, "_datasets_pool"):
            df = pd.read_csv("%s/pool_init.csv" % self.save_dir)
            self._datasets_pool = [get_subset_from_uidx(dataset, id2datapoint, df["uidx"].tolist()) 
                                   for dataset, id2datapoint in zip(self.datasets_empty, self.id2datapoints)]
        return self._datasets_pool

    @property
    def datasets_val(self):
        if not hasattr(self, "_datasets_val"):
            df = pd.read_csv("%s/val.csv" % self.save_dir)
            self._datasets_val = [get_subset_from_uidx(dataset, id2datapoint, df["uidx"].tolist()) 
                                  for dataset, id2datapoint in zip(self.datasets_empty, self.id2datapoints)]
        return self._datasets_val
    
    def process_args(self) -> None:
        super().process_args()


class SelectorArgs(Tap):
    select_method: Literal["random", "explorative", "exploitive"] = None
    """the method to select the next batch of samples."""
    s_batch_size: int = 1
    """number of samples to select in each iteration."""
    s_batch_mode: Literal["naive", "cluster"] = "naive"
    """the method to select the next batch of samples."""
    s_cluster_size: int = None
    """number of samples in each cluster for cluster batch selection. (default = 20 * batch_size)"""
    s_query_size: int = None
    """number of samples to query in each active learning iteration. (default=None means query all samples in the 
    pool set)"""
    s_exploitive_target: str = None
    """the target value for exploitive active learning."""
    seed: int = 0
    """random seed."""

    @property
    def selector(self):
        if not hasattr(self, "_selector"):
            if self.s_exploitive_target is not None and self.s_exploitive_target not in ["min", "max"]:
                self.s_exploitive_target = float(self.s_exploitive_target)
            if self.select_method == "random":
                if self.s_batch_mode == "cluster":
                    self._selector = ClusterRandomSelector(batch_size=self.s_batch_size,
                                                           cluster_size=self.s_cluster_size,
                                                           seed=self.seed)
                else:
                    self._selector = RandomSelector(batch_size=self.s_batch_size, seed=self.seed)
            elif self.select_method == "explorative":
                if self.s_batch_mode == "cluster":
                    if self.s_query_size is not None:
                        self._selector = PartialQueryClusterExplorativeSelector(
                            batch_size=self.s_batch_size,
                            cluster_size=self.s_cluster_size,
                            query_size=self.s_query_size,
                            seed=self.seed)
                    else:
                        self._selector = ClusterExplorativeSelector(batch_size=self.s_batch_size,
                                                                    cluster_size=self.s_cluster_size,
                                                                    seed=self.seed)
                else:
                    if self.s_query_size is not None:
                        self._selector = PartialQueryExplorativeSelector(batch_size=self.s_batch_size,
                                                                         query_size=self.s_query_size,
                                                                         seed=self.seed)
                    else:
                        self._selector = ExplorativeSelector(batch_size=self.s_batch_size, seed=self.seed)
            elif self.select_method == "exploitive":
                if self.s_batch_mode == "cluster":
                    if self.s_query_size is not None:
                        self._selector = PartialQueryClusterExploitiveSelector(
                            batch_size=self.s_batch_size,
                            cluster_size=self.s_cluster_size,
                            query_size=self.s_query_size,
                            target=self.s_exploitive_target,
                            seed=self.seed)
                    else:
                        self._selector = ClusterExploitiveSelector(batch_size=self.s_batch_size,
                                                                   cluster_size=self.s_cluster_size,
                                                                   target=self.s_exploitive_target,
                                                                   seed=self.seed)
                else:
                    if self.s_query_size is not None:
                        self._selector = PartialQueryExploitiveSelector(batch_size=self.s_batch_size, 
                                                                        query_size=self.s_query_size,
                                                                        target=self.s_exploitive_target,
                                                                        seed=self.seed)
                    else:
                        self._selector = ExploitiveSelector(batch_size=self.s_batch_size, 
                                                            target=self.s_exploitive_target, 
                                                            seed=self.seed)
            elif self.select_method is None:
                return None
            else:
                raise ValueError(f"Unknown selection method {self.select_method}")
        return self._selector


class ForgetterArgs(Tap):
    forget_method: Literal["first", "random", 
                           "min_oob_uncertainty", "max_oob_uncertainty", 
                           "min_oob_error", "max_oob_error",
                           "min_loo_uncertainty", "max_loo_uncertainty",
                           "min_loo_error", "max_loo_error"] = None
    """the forget method."""
    f_batch_size: int = 1
    """number of samples to forget in each iteration."""
    f_min_train_size: int = None
    """Forget method activate only when the number of samples in the training set larger than this number."""
    seed: int = 0
    """random seed."""

    @property
    def forgetter(self):
        if not hasattr(self, "_forgetter"):
            if self.forget_method == "first":
                self._forgetter = FirstForgetter(batch_size=self.f_batch_size)
            elif self.forget_method == "random":
                self._forgetter = RandomForgetter(batch_size=self.f_batch_size, seed=self.seed)
            elif self.forget_method == "min_oob_uncertainty":
                self._forgetter = MinOOBUncertaintyForgetter(batch_size=self.f_batch_size, seed=self.seed)
            elif self.forget_method == "max_oob_uncertainty":
                self._forgetter = MaxOOBUncertaintyForgetter(batch_size=self.f_batch_size, seed=self.seed)
            elif self.forget_method == "min_oob_error":
                self._forgetter = MinOOBErrorForgetter(batch_size=self.f_batch_size, seed=self.seed)
            elif self.forget_method == "max_oob_error":
                self._forgetter = MaxOOBErrorForgetter(batch_size=self.f_batch_size, seed=self.seed)
            elif self.forget_method == "min_loo_uncertainty":
                self._forgetter = MinLOOUncertaintyForgetter(batch_size=self.f_batch_size, seed=self.seed)
            elif self.forget_method == "max_loo_uncertainty":
                self._forgetter = MaxLOOUncertaintyForgetter(batch_size=self.f_batch_size, seed=self.seed)
            elif self.forget_method == "min_loo_error":
                self._forgetter = MinLOOErrorForgetter(batch_size=self.f_batch_size, seed=self.seed)
            elif self.forget_method == "max_loo_error":
                self._forgetter = MaxLOOErrorForgetter(batch_size=self.f_batch_size, seed=self.seed)
            elif self.forget_method is None:
                return None
            else:
                raise ValueError(f"Unknown forget method {self.forget_method}")
        return self._forgetter


class EvaluationArgs(Tap):
    metrics: List[Metric] = None
    """the metrics to evaluate model performance."""
    evaluate_stride: int = None
    """evaluate model performance on the validation set when the size of the training set is an integer multiple of the 
    evaluation stride."""
    top_k: float = None
    """ For regression tasks, the ratio of top molecules are considered.
        For binary classification tasks, the molecules with label 1 will considered as top molecules.
    """
    detail: bool = False
    """output the details of each active learning iteration."""


class LearningArgs(DatasetModelArgs, SelectorArgs, ForgetterArgs, EvaluationArgs):
    n_select: int = 1
    """number of selecting steps in each select-forget loop."""
    n_forget: int = 0
    """number of forgetting steps in each select-forget loop."""
    max_iter: int = None
    """the maximum number of select-forget loops."""
    save_cpt_stride: int = None
    """save checkpoint file every no. steps of select-forget loops."""
    write_traj_stride: int = 1
    """write trajectory file every no. steps of select-forget loops."""
    load_checkpoint: bool = False
    """load checkpoint file and continue the active learning."""

    @property
    def top_uidx(self) -> Optional[List[int]]:
        if self.top_k is not None:
            top_uidx = []
            if self.task_type == "binary":
                for data in (self.datasets_train[0].data + self.datasets_pool[0].data):
                    if data.targets[0] == 1:
                        top_uidx.append(data.uidx)
            else:
                assert 0. < self.top_k < 1., "top_k must be in (0, 1)."
                n_top_k = math.ceil(self.top_k * (len(self.datasets_train[0]) + len(self.datasets_pool[0])))
                y_AL = self.datasets_train[0].y.ravel().tolist() + self.datasets_pool[0].y.ravel().tolist()
                top_k_index = get_topn_idx(y_AL, n_top_k, target=self.s_exploitive_target)
                for i, data in enumerate(self.datasets_train[0].data + self.datasets_pool[0].data):
                    if i in top_k_index:
                        top_uidx.append(data.uidx)
            return top_uidx
        else:
            return None
        
    def process_args(self) -> None:
        super().process_args()
        for i, model_config_dict in enumerate(self.model_configs_dict):
            if model_config_dict.get("graph_kernel_type") == "graph":
                self.datasets_full[i].unify_datatype()
        # check the input for exploitive active learning
        if self.select_method == "exploitive":
            # assert self.task_type == "regression", "exploitive active learning only support regression task."
            # assert self.top_k is not None, "top_k must be set for exploitive active learning."
            assert self.s_exploitive_target is not None, "s_exploitive_target must be set for exploitive active learning."

        if self.select_method is not None:
            assert self.n_select > 0, "n_select must be greater than 0."
        if self.forget_method is not None:
            assert self.n_forget > 0, "n_forget must be greater than 0."

        if self.max_iter is None:
            dn = self.n_select - self.n_forget
            if dn > 0:
                self.max_iter = len(self.datasets_pool[0]) // dn
            elif dn < 0:
                self.max_iter = len(self.datasets_train[0]) // (-dn) - 1
            else:
                self.max_iter = len(self.datasets_pool[0]) // self.n_select
