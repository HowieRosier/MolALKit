#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Union, Literal
import pickle
import os
from logging import Logger
from sklearn.gaussian_process.kernels import RBF, DotProduct
from mgktools.data.data import Dataset
from mgktools.kernels.utils import get_kernel_config
from mgktools.kernels.PreComputed import calc_precomputed_kernel_config
from molalkit.exe.logging import EmptyLogger


def get_model(data_format: Literal["mgktools", "chemprop", "graphgps"],
              task_type: Literal["regression", "binary", "multiclass"],
              n_jobs: int = 8,
              seed: int = 0,
              # arguments for classical machine learning models
              model: Literal["random_forest", "naive_bayes", "logistic_regression", "gaussian_process_regression",
                             "gaussian_process_classification", "support_vector_machine", "adaboost", "xgboost", 
                             "decision_tree", "extra_trees", "MultinomialNB", "BernoulliNB", "GaussianNB", 
                             "LSTM", "GRU", "MolFormer"] = "random_forest",
              kernel=None,
              uncertainty_type: Literal["value", "uncertainty"] = None,
              alpha: Union[float, str] = 1e-8,
              C: float = 1.0,
              booster: Literal["gbtree", "gblinear", "dart"] = "gbtree",
              n_estimators: int = 100,
              max_depth: int = None,
              learning_rate: float = 0.1,
              tokenizer: Literal["SMILES", "SELFIES"] = None,
              smiles_full: List[str] = None,
              embedding_size: int = 64,
              # chemprop arguments
              save_dir: str = None,
              data_path: str = None,
              smiles_columns: List[str] = None,
              target_columns: List[str] = None,
              loss_function: Literal["mse", "bounded_mse", "binary_cross_entropy", "cross_entropy", "mcc", "sid",
                                     "wasserstein", "mve", "evidential", "dirichlet"] = None,
              multiclass_num_classes: int = 3,
              features_generator=None,
              no_features_scaling: bool = False,
              features_only: bool = False,
              features_size: int = 0,
              epochs: int = 30,
              depth: int = 3,
              hidden_size: int = 300,
              ffn_num_layers: int = 2,
              ffn_hidden_size: int = None,
              dropout: float = 0.0,
              batch_size: int = 50,
              ensemble_size: int = 1,
              number_of_molecules: int = 1,
              mpn_shared: bool = False,
              atom_messages: bool = False,
              undirected: bool = False,
              class_balance: bool = False,
              checkpoint_dir: str = None,
              checkpoint_frzn: str = None,
              frzn_ffn_layers: int = 0,
              freeze_first_only: bool = False,
              mpn_path: str = None,
              freeze_mpn: bool = False,
              uncertainty_method: Literal["mve", "ensemble", "evidential_epistemic", "evidential_aleatoric",
                                          "evidential_total", "classification", "dropout", "spectra_roundrobin"] = None,
              uncertainty_dropout_p: float = 0.1,
              dropout_sampling_size: int = 10,
              continuous_fit: bool = False,
              # L2 regularization
              weight_decay: float = 0.0,
              perturb_sigma: float = 0.0,
              # CBP (Continual Backpropagation) parameters
              cbp: bool = False,
              maturity_threshold: int = 20,
              replacement_rate: float = 1e-4,
              decay_rate: float = 0.99,
              util_type: str = 'contribution',
              # graphgps arguments
              cfg_path: str = None,
              # MolFormer arguments
              pretrained_path: str = None,
              n_head: int = 12,
              n_layer: int = 12,
              n_embd: int = 768,
              d_dropout: float = 0.1,
              num_feats: int = 32,
              logger: Logger = None):
    if alpha.__class__ == str:
        alpha = float(open(alpha).read())

    if data_format == "mgktools":
        if model == "random_forest":
            if task_type == "regression":
                from molalkit.models.random_forest.RandomForestRegressor import RFRegressor
                return RFRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, random_state=seed)
            else:
                from molalkit.models.random_forest.RandomForestClassifier import RFClassifier
                return RFClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, random_state=seed, oob_score=True)
        elif model == "MultinomialNB":
            assert task_type == "binary"
            from molalkit.models.naive_bayes.NaiveBayesClassifier import MultinomialNBClassifier
            return MultinomialNBClassifier()
        elif model == "BernoulliNB":
            assert task_type == "binary"
            from molalkit.models.naive_bayes.NaiveBayesClassifier import BernoulliNBClassifier
            return BernoulliNBClassifier()
        elif model == "GaussianNB":
            assert task_type == "binary"
            from molalkit.models.naive_bayes.NaiveBayesClassifier import GaussianNBClassifier
            return GaussianNBClassifier()
        elif model == "logistic_regression":
            assert task_type == "binary"
            from molalkit.models.logistic_regression.LogisticRegression import LogisticRegressor
            return LogisticRegressor(random_state=seed)
        elif model == "decision_tree":
            assert task_type == "binary"
            from molalkit.models.dt.dt import DecisionTreeClassifier
            return DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
        elif model == "extra_trees":
            assert task_type == "binary"
            from molalkit.models.extra_trees.extra_trees import ExtraTreesClassifier
            return ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, random_state=seed)
        elif model == "gaussian_process_regression":
            assert task_type in ["regression", "binary"]
            assert uncertainty_type is not None
            from molalkit.models.gaussian_process.GaussianProcessRegressor import GPRegressor
            return GPRegressor(kernel=kernel, alpha=alpha, optimizer=None, uncertainty_type=uncertainty_type)
        elif model == "gaussian_process_classification":
            assert task_type == "binary"
            from molalkit.models.gaussian_process.GaussianProcessClassifier import GPClassifier
            return GPClassifier(kernel=kernel, optimizer=None)
        elif model == "support_vector_machine":
            if task_type == "regression":
                from molalkit.models.support_vector.SupportVectorRegressor import SVRegressor
                return SVRegressor(kernel=kernel, C=C)
            else:
                from molalkit.models.support_vector.SupportVectorClassifier import SVClassifier
                return SVClassifier(kernel=kernel, C=C, probability=True)
        elif model == "adaboost":
            if task_type == "regression":
                from molalkit.models.adaboost.AdaBoostRegressor import AdaBoostRegressor
                return AdaBoostRegressor(random_state=seed)
            else:
                from molalkit.models.adaboost.AdaBoostClassifier import AdaBoostClassifier
                return AdaBoostClassifier(random_state=seed)
        elif model == "xgboost":
            if task_type == "regression":
                from molalkit.models.xgboost.XGBRegressor import XGBRegressor
                return XGBRegressor(booster=booster, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, n_jobs=n_jobs, random_state=seed)
            else:
                from molalkit.models.xgboost.XGBClassifier import XGBClassifier
                return XGBClassifier(booster=booster, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, n_jobs=n_jobs, random_state=seed)
        elif model == "gradient_boosting":
            if task_type == "regression":
                from molalkit.models.gradient_boosting.gradient_boosting import GradientBoostingRegressor
                return GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=seed)
            else:
                from molalkit.models.gradient_boosting.gradient_boosting import GradientBoostingClassifier
                return GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=seed)
        elif model in ["LSTM", "GRU"]:
            from molalkit.models.rnn.rnn import RNN
            from molalkit.models.rnn.tokenizer import SMILESTokenizer, SELFIESTokenizer
            if tokenizer == "SMILES":
                tokenizer_ = SMILESTokenizer()
            elif tokenizer == "SELFIES":
                tokenizer_ = SELFIESTokenizer()
            else:
                raise ValueError(f"unknown tokenizer: {tokenizer}")
            tokenizer_.create_vocabulary(smiles_full)
            return RNN(tokenizer=tokenizer_,
                       task_type=task_type,
                       rnn_type=model,
                       embedding_size=embedding_size,
                       depth=depth,
                       hidden_size=hidden_size,
                       dropout=dropout,
                       epochs=epochs,
                       ffn_num_layers=ffn_num_layers,
                       batch_size=batch_size)
        elif model == "MolFormer":
            from molalkit.models.molformer.molformer import MolFormer
            return MolFormer(save_dir=save_dir, task_type=task_type, pretrained_path=pretrained_path,
                             num_tasks=len(target_columns),
                             n_head=n_head, n_layer=n_layer, n_embd=n_embd, d_dropout=d_dropout,
                             dropout=dropout, learning_rate=learning_rate, 
                             num_feats=num_feats, ensemble_size=ensemble_size, epochs=epochs,
                             batch_size=batch_size, num_workers=n_jobs, seed=seed)
        else:
            raise ValueError(f"unknown model: {model}")
    elif data_format == "chemprop":
        from molalkit.models.mpnn.mpnn import MPNN
        return MPNN(save_dir=save_dir,
                    data_path=data_path,
                    smiles_columns=smiles_columns,
                    target_columns=target_columns,
                    dataset_type="classification" if task_type == "binary" else task_type,
                    loss_function=loss_function,
                    multiclass_num_classes=multiclass_num_classes,
                    features_generator=features_generator,
                    no_features_scaling=no_features_scaling,
                    features_only=features_only,
                    features_size=features_size,
                    epochs=epochs,
                    depth=depth,
                    hidden_size=hidden_size,
                    ffn_num_layers=ffn_num_layers,
                    ffn_hidden_size=ffn_hidden_size,
                    dropout=dropout,
                    batch_size=batch_size,
                    ensemble_size=ensemble_size,
                    number_of_molecules=number_of_molecules,
                    mpn_shared=mpn_shared,
                    atom_messages=atom_messages,
                    undirected=undirected,
                    class_balance=class_balance,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_frzn=checkpoint_frzn,
                    frzn_ffn_layers=frzn_ffn_layers,
                    freeze_first_only=freeze_first_only,
                    mpn_path=mpn_path,
                    freeze_mpn=freeze_mpn,
                    uncertainty_method=uncertainty_method,
                    uncertainty_dropout_p=uncertainty_dropout_p,
                    dropout_sampling_size=dropout_sampling_size,
                    weight_decay=weight_decay,
                    perturb_sigma=perturb_sigma,
                    n_jobs=n_jobs,
                    seed=seed,
                    continuous_fit=continuous_fit,
                    # CBP parameters
                    cbp=cbp,
                    maturity_threshold=maturity_threshold,
                    replacement_rate=replacement_rate,
                    decay_rate=decay_rate,
                    util_type=util_type,
                    logger=logger or EmptyLogger())
    elif data_format == "graphgps":
        from molalkit.models.graphgps.graphgps import GraphGPS
        return GraphGPS(save_dir=save_dir,
                        cfg_path=cfg_path,
                        ensemble_size=ensemble_size,
                        number_of_molecules=number_of_molecules,
                        n_jobs=n_jobs,
                        seed=seed)
    else:
        raise ValueError(f"unknown data_format {data_format}")


def get_kernel(graph_kernel_type: Literal["graph", "pre-computed", "no"] = "no",
               mgk_files: List[str] = None,
               features_kernel_type: Literal["dot_product", "rbf"] = None,
               features_hyperparameters: Union[float, List[float]] = None,
               features_hyperparameters_file: str = None,
               dataset: Dataset = None,
               kernel_pkl_path: str = None,
               ):
    if mgk_files is None:
        assert graph_kernel_type == "no"
        # no graph kernel involved.
        if features_kernel_type is None:
            return None
        elif features_kernel_type == "linear":
            raise NotImplementedError
        elif features_kernel_type == "dot_product":
            if features_hyperparameters.__class__ == list:
                assert len(features_hyperparameters) == 1
                sigma_0 = features_hyperparameters[0]
            else:
                sigma_0 = features_hyperparameters
            return DotProduct(sigma_0=sigma_0)
        elif features_kernel_type == "rbf":
            return RBF(length_scale=features_hyperparameters)
        else:
            raise ValueError
    else:
        if graph_kernel_type == "graph":
            return get_kernel_config(
                dataset=dataset,
                graph_kernel_type="graph",
                mgk_hyperparameters_files=mgk_files,
                features_kernel_type=features_kernel_type,
                features_hyperparameters=features_hyperparameters,
                features_hyperparameters_bounds="fixed",
                features_hyperparameters_file=features_hyperparameters_file
            ).kernel
        elif graph_kernel_type == "pre-computed":
            assert kernel_pkl_path is not None
            if os.path.exists(kernel_pkl_path):
                return get_kernel_config(
                    dataset=dataset,
                    graph_kernel_type="pre-computed",
                    features_kernel_type=features_kernel_type,
                    features_hyperparameters=features_hyperparameters,
                    features_hyperparameters_bounds="fixed",
                    features_hyperparameters_file=features_hyperparameters_file,
                    kernel_pkl=kernel_pkl_path
                ).kernel
            else:
                dataset.graph_kernel_type = "graph"
                kernel_config = get_kernel_config(
                    dataset=dataset,
                    graph_kernel_type="graph",
                    mgk_hyperparameters_files=mgk_files,
                    features_kernel_type=features_kernel_type,
                    features_hyperparameters=features_hyperparameters,
                    features_hyperparameters_bounds="fixed",
                    features_hyperparameters_file=features_hyperparameters_file
                )
                kernel_config = calc_precomputed_kernel_config(kernel_config=kernel_config, dataset=dataset)
                dataset.graph_kernel_type = "pre-computed"
                pickle.dump(kernel_config, open(kernel_pkl_path, "wb"), protocol=4)
                return kernel_config.kernel
        else:
            raise ValueError
