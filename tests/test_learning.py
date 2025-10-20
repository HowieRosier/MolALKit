#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
import shutil
import numpy as np
from molalkit.exe.run import molalkit_run


CWD = os.path.dirname(os.path.abspath(__file__))


MODEL_CONFIGS = [
    "RandomForest_Morgan_Config",
    "RandomForest_RDKitNorm_Config",
    "RandomForest_RDKitNorm_Morgan_Config",
    "AdaBoost_RDKitNorm_Config",
    "GaussianProcessRegressionPosteriorUncertainty_MarginalizedGraphKernel_Config",
    "GaussianProcessRegressionPosteriorUncertainty_RBFKernelRDKitNorm_Config",
    "GaussianProcessRegressionPosteriorUncertainty_MarginalizedGraphKernel+RBFKernelRDKitNorm_Config",
    "SupportVectorMachine_MarginalizedGraphKernel_Config",
    "SupportVectorMachine_RBFKernelRDKitNorm_Config",
    "SupportVectorMachine_MarginalizedGraphKernel+RBFKernelRDKitNorm_Config",
    "XGBoost_Morgan_Config"
]
MODEL_CONFIGS_REGRESSION = [
    "MLP_RDKitNorm_Regression_Config",
    "MLP_RDKitNorm_Regression_MVE_Config",
    "MLP_RDKitNorm_Regression_Evidential_Config",
    "DMPNN_Regression_Config",
    "DMPNN_Regression_MVE_Config",
    "DMPNN_Regression_Evidential_Config",
    "DMPNN+RDKitNorm_Regression_Config",
    "DMPNN+RDKitNorm_Regression_MVE_Config",
    "DMPNN+RDKitNorm_Regression_Evidential_Config",
]
MODELS_CONFIGS_CLASSIFICATION = [
    "BernoulliNB_RDKitNorm_Config",
    "GaussianNB_Morgan_Config",
    "MultinomialNB_RDKitNorm_Config",
    "LogisticRegression_Morgan_Config",
    "GaussianProcessClassification_MarginalizedGraphKernel_Config",
    "GaussianProcessClassification_RBFKernelRDKitNorm_Config",
    "GaussianProcessClassification_MarginalizedGraphKernel+RBFKernelRDKitNorm_Config",
    "GaussianProcessRegressionDecisionBoundaryUncertainty_MarginalizedGraphKernel_Config",
    "GaussianProcessRegressionDecisionBoundaryUncertainty_RBFKernelRDKitNorm_Config",
    "GaussianProcessRegressionDecisionBoundaryUncertainty_MarginalizedGraphKernel+RBFKernelRDKitNorm_Config",
    "MLP_Morgan_BinaryClassification_Config",
    "DMPNN_BinaryClassification_Config",
    "DMPNN+RDKitNorm_BinaryClassification_Config",
]


@pytest.mark.parametrize("input1", [
    ("freesolv", "regression"),
    ("dili",  "binary"),
])
@pytest.mark.parametrize("split_type", ["random", "scaffold_random", "scaffold_order"])
@pytest.mark.parametrize("split_sizes", [["0.5", "0.5"], ["1.0", "0.0"]])
@pytest.mark.parametrize("error_rate", [None, "0.1"])
@pytest.mark.parametrize("input2", [
    ("2", "1", "0"),
    ("10000", "0", "1"),
])
@pytest.mark.parametrize("model_config1", ["RandomForest_RDKitNorm_Config", 
                                           "GaussianProcessRegressionPosteriorUncertainty_RBFKernelRDKitNorm_Config"])
@pytest.mark.parametrize("model_config2", [None])
@pytest.mark.parametrize("select_method", [None, "random", "explorative", "exploitive"])
@pytest.mark.parametrize("s_batch_size", ["1", "10"])
@pytest.mark.parametrize("s_batch_mode", ["naive", "cluster"])
@pytest.mark.parametrize("s_exploitive_target", [None, "min", "max", "-4"])
@pytest.mark.parametrize("forget_method", [None, "first", "random", 
                         "min_oob_uncertainty", "max_oob_uncertainty", 
                         "min_oob_error", "max_loo_error",
                         "min_loo_uncertainty", "max_loo_uncertainty",
                         "min_loo_error", "max_loo_error"])
@pytest.mark.parametrize("f_batch_size", ["1", "10"])
@pytest.mark.parametrize("evaluate_stride", [None, "1", "3"])
@pytest.mark.parametrize("top_k", [None, "0.1"])
@pytest.mark.parametrize("detail", [True, False])
@pytest.mark.parametrize("max_iter", [None, "50"])
def test_MolALKitRun(input1, split_type, split_sizes, error_rate, input2,
                      model_config1, model_config2,
                      select_method, s_batch_size, s_batch_mode, s_exploitive_target,
                      forget_method, f_batch_size,
                      evaluate_stride, top_k, detail, max_iter):
    if np.random.uniform(0, 1) > 0.01:
        return
    data_public, task_type = input1
    init_size, n_select, n_forget = input2
    model_configs = [model_config1]
    if model_config2 is not None:
        model_configs.append(model_config2)
    metrics = ["rmse", "r2"] if task_type == "regression" else ["roc_auc", "mcc"]

    ### skip the invalid input combinations
    if task_type == "regression":
        if model_config1 in MODELS_CONFIGS_CLASSIFICATION or model_config2 in MODELS_CONFIGS_CLASSIFICATION:
            return
        if error_rate is not None:
            return
    elif task_type == "binary":
        if model_config1 in MODEL_CONFIGS_REGRESSION or model_config2 in MODEL_CONFIGS_REGRESSION:
            return
    if n_select == "0" and select_method is not None:
        return
    if s_batch_size == "1" and s_batch_mode == "cluster":
        return
    if s_exploitive_target is not None and select_method != "exploitive":
        return
    if forget_method is not None:
        if n_forget == "0":
            return
        if "oob" in forget_method and not model_config1.startswith("RandomForest"):
            return
        if "loo" in forget_method and not model_config1.startswith("GaussianProcessRegression"):
            return

    arguments = [
        "--save_dir", f"{CWD}/tmp",
        "--data_public", data_public,
        "--metrics", *metrics,
        "--split_type", split_type,
        "--split_sizes", *split_sizes,
        "--init_size", init_size,
        "--model_configs", *model_configs,
        "--n_select", n_select,
        "--n_forget", n_forget,
    ]
    if error_rate is not None:
        arguments.extend(["--error_rate", error_rate])
    if select_method is not None:
        arguments.extend(["--select_method", select_method,
                          "--s_batch_size", s_batch_size,
                          "--s_batch_mode", s_batch_mode])
    if s_exploitive_target is not None:
        arguments.extend(["--s_exploitive_target", s_exploitive_target])
    if forget_method is not None:
        arguments.extend(["--forget_method", forget_method,
                          "--f_batch_size", f_batch_size])
    if evaluate_stride is not None:
        arguments.extend(["--evaluate_stride", evaluate_stride])
    if top_k is not None:
        arguments.extend(["--top_k", top_k])
    if detail:
        arguments.append("--detail")
    if max_iter is not None:
        arguments.extend(["--max_iter", max_iter])
    molalkit_run(arguments)
    shutil.rmtree(f"{CWD}/tmp")
