#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Union, Literal
import numpy as np
from mgktools.evaluators.metric import AVAILABLE_METRICS_BINARY, AVAILABLE_METRICS_REGRESSION, metric_binary, metric_regression


def eval_metric_func(y, y_pred, metric: str) -> float:
    if metric in AVAILABLE_METRICS_REGRESSION:
        return metric_regression(y, y_pred, metric)
    elif metric in AVAILABLE_METRICS_BINARY:
        return metric_binary(y, y_pred, metric)
    else:
        raise RuntimeError(f"Unsupported metrics {metric}")


def random_choice(N, size):
    if size < N:
        idx = np.random.choice(N, size, replace=False).tolist()
        mask = np.isin(range(N), idx)
        idx_remain = np.arange(N)[~mask].tolist()
    else:
        idx = list(range(N))
        idx_remain = []
    return idx, idx_remain


def get_topn_idx(values: np.ndarray, n: int = 1, target: Union[Literal["max", "min"], float] = "max",
                 cutoff: float = None) -> List[int]:
    """ Get the indices of top n values.

    Parameters
    ----------
    values: array-like.
    n: number of indices to be selected.
    target: "max", "min", or a float value.
    cutoff: if not None, only values >= cutoff (when target=max) will be considered.

    Returns
    -------

    """
    if isinstance(values, list):
        values = np.array(values)
    if target == "min":
        values = - values
        if cutoff is not None:
            cutoff = - cutoff
    elif isinstance(target, float):
        assert cutoff is None
        values = - np.absolute(values - target)  # distance from target
    if cutoff is not None:
        n_candidates = len(np.where(values >= cutoff)[0])
        n = min(n, n_candidates)
        if n == 0:
            return []
    # Includes tiny random values to randomly sort duplicated values
    sorting_key = values + np.random.random(len(values)) * 1e-10
    return np.argsort(sorting_key)[-n:].tolist()
