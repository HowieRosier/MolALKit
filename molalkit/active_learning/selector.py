#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
from molalkit.active_learning.utils import random_choice, get_topn_idx
from molalkit.data.utils import get_subset_from_idx


class BaseSelector(ABC):
    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size

    @abstractmethod
    def __call__(self, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        pass

    @property
    @abstractmethod
    def info(self) -> str:
        pass


class BaseRandomSelector(BaseSelector, ABC):
    """Base Selector that uses random seed."""
    def __init__(self, batch_size: int = 1, seed: int = 0):
        super().__init__(batch_size=batch_size)
        self.seed = seed
        np.random.seed(seed)


class BaseClusterSelector(BaseRandomSelector, ABC):
    """ Base Selector that uses clustering method.
        cluster_size samples will be selected first according to provided selection algorithm,
        and then separate into batch_size clusters using KMeans clustering method.
        Finally, one sample from each cluster will be selected.
        This can increase the diversity of the selected samples batch.
    """
    def __init__(self, batch_size: int = 1, seed: int = 0, cluster_size: int = None):
        super().__init__(batch_size=batch_size, seed=seed)
        assert batch_size > 1, "batch_size should be larger than 1 for cluster selection method."
        if cluster_size is None:
            self.cluster_size = batch_size * 20
        else:
            assert cluster_size >= batch_size, "cluster_size should be larger than batch_size."
            self.cluster_size = cluster_size

    def get_idx_cluster(self, dataset_pool, kernel: Callable, idx_cluster) -> List[int]:
        """ Find distant samples from a pool using KMeans clustering method."""
        if len(idx_cluster) < self.batch_size:
            return list(range(len(idx_cluster)))
        else:
            K = kernel(dataset_pool.X[idx_cluster])
            add_idx = self.find_distant_samples(gram_matrix=K, batch_size=self.batch_size)
            return add_idx

    def find_distant_samples(self, gram_matrix: np.ndarray[float], batch_size: int = 1) -> List[int]:
        """ Find distant samples from a pool using clustering method.

        Parameters
        ----------
        gram_matrix: gram (kernel) matrix of the samples.
        batch_size: number of samples to be selected.

        Returns
        -------
        List of idx
        """
        embedding = SpectralEmbedding(
            n_components=batch_size,
            affinity="precomputed",
            random_state=self.seed
        ).fit_transform(gram_matrix)

        cluster_result = KMeans(
            n_clusters=batch_size,
            random_state=self.seed
        ).fit_predict(embedding)
       # Calculate cluster centers
        centers = np.array([embedding[cluster_result == i].mean(axis=0)
                           for i in range(batch_size)])
        # For each cluster, find the sample that's most distant from other cluster centers
        total_distance = defaultdict(dict)
        for i in range(batch_size):
            cluster_idx = cluster_result[i]
            # Get centers of other clusters
            other_centers = np.delete(centers, cluster_idx, axis=0)
            # Calculate sum of inverse distances to other centers
            # (Using inverse distance means smaller values indicate more distant points)
            sum_inverse_distances = np.sum(np.sqrt(np.sum(
                np.square(embedding[i] - other_centers), axis=1
            )) ** -0.5)
            total_distance[cluster_idx][sum_inverse_distances] = i
        # Select the most distant sample from each cluster
        selected_indices = [
            total_distance[i][min(total_distance[i].keys())]
            for i in range(batch_size)
        ]
        return selected_indices


class BasePartialQuerySelector(BaseRandomSelector, ABC):
    """ Base Selector that uses partial query method.
        query_size samples will be selected first randomly,
        and then batch_size samples will be selected according to provided selection algorithm.
    """
    @staticmethod
    def get_partial_data(dataset_pool, query_size):
        # randomly select query_size samples from dataset_pool.
        if len(dataset_pool) <= query_size:
            return dataset_pool, np.array(range(len(dataset_pool)))
        else:
            idx_query = np.random.choice(range(len(dataset_pool)), query_size, replace=False)
            data_query = get_subset_from_idx(dataset_pool, idx_query)
            return data_query, idx_query


class RandomSelector(BaseRandomSelector):
    def __call__(self, dataset_pool, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        idx, idx_remain = random_choice(len(dataset_pool), self.batch_size)
        return idx, [], idx_remain

    @property
    def info(self) -> str:
        return f"RandomSelector(batch_size={self.batch_size})"


class ClusterRandomSelector(BaseClusterSelector):
    def __call__(self, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size < len(dataset_pool):
            idx_cluster = random_choice(len(dataset_pool), self.cluster_size)[0]
            idx = idx_cluster[np.array(self.get_idx_cluster(dataset_pool, kernel, idx_cluster))].tolist()

            mask = np.isin(range(len(dataset_pool)), idx)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, [], idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"ClusterRandomSelector(batch_size={self.batch_size}, cluster_size={self.cluster_size})"


class ExplorativeSelector(BaseRandomSelector):
    def __call__(self, model, dataset_pool, stop_cutoff: float = None, confidence_cutoff: float = None, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            y_std = model.predict_uncertainty(dataset_pool)
            idx = get_topn_idx(y_std, n=self.batch_size, target="max", cutoff=stop_cutoff)

            acquisition = y_std[np.array(idx)].tolist()

            idx_confident = np.where(y_std < confidence_cutoff)[0].tolist() if confidence_cutoff is not None else []
            mask = np.isin(range(len(dataset_pool)), idx + idx_confident)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"ExplorativeSelector(batch_size={self.batch_size})"


class ClusterExplorativeSelector(BaseClusterSelector):
    def __call__(self, model, dataset_pool, kernel: Callable, stop_cutoff: float = None, confidence_cutoff: float = None, 
                 **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            y_std = model.predict_uncertainty(dataset_pool)
            idx_cluster = get_topn_idx(y_std, n=self.cluster_size, target="max", cutoff=stop_cutoff)
            idx = np.array(idx_cluster)[np.array(self.get_idx_cluster(dataset_pool, kernel, idx_cluster))].tolist()

            acquisition = y_std[np.array(idx)].tolist()

            idx_confident = np.where(y_std < confidence_cutoff)[0].tolist() if confidence_cutoff is not None else []
            mask = np.isin(range(len(dataset_pool)), idx + idx_confident)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"ClusterExplorativeSelector(batch_size={self.batch_size}, cluster_size={self.cluster_size})"


class PartialQueryExplorativeSelector(BasePartialQuerySelector, BaseRandomSelector):
    def __init__(self, query_size: int, batch_size: int = 1, seed: int = 0):
        super().__init__(batch_size=batch_size, seed=seed)
        self.query_size = query_size
        assert batch_size <= query_size, "batch_size should be smaller than query_size."

    def __call__(self, model, dataset_pool, stop_cutoff: float = None, confidence_cutoff: float = None, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            data_query, idx_query = self.get_partial_data(dataset_pool, self.query_size)
            y_std = model.predict_uncertainty(data_query)
            idx_ = get_topn_idx(y_std, n=self.batch_size, target="max", cutoff=stop_cutoff)
            idx = idx_query[np.array(idx_)].tolist()

            acquisition = y_std[idx_].tolist()

            idx_confident_ = np.where(y_std < confidence_cutoff)[0].tolist() if confidence_cutoff is not None else []
            idx_confident = idx_query[np.array(idx_confident_)].tolist()
            mask = np.isin(range(len(dataset_pool)), idx + idx_confident)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return f"PartialQueryExplorativeSelector(batch_size={self.batch_size}, query_size={self.query_size})"


class PartialQueryClusterExplorativeSelector(BaseClusterSelector, BasePartialQuerySelector):
    def __init__(self, query_size: int, batch_size: int = 1, cluster_size: int = None, seed: int = 0):
        super().__init__(batch_size=batch_size, cluster_size=cluster_size, seed=seed)
        self.query_size = query_size
        assert self.cluster_size <= query_size, "cluster_size should be smaller than query_size."

    def __call__(self, model, dataset_pool, kernel: Callable, stop_cutoff: float = None, confidence_cutoff: float = None, 
                 **kwargs) -> Tuple[List[int], List[float], List[int]]:
        if self.batch_size <= len(dataset_pool):
            data_query, idx_query = self.get_partial_data(dataset_pool, self.query_size)
            y_std = model.predict_uncertainty(data_query)
            idx_cluster = get_topn_idx(y_std, n=self.cluster_size, target="max", cutoff=stop_cutoff)
            idx_ = np.array(idx_cluster)[np.array(self.get_idx_cluster(dataset_pool, kernel, idx_cluster))].tolist()
            idx = np.array(idx_query)[np.array(idx_)].tolist()

            acquisition = y_std[np.array(idx_)].tolist()

            idx_confident_ = np.where(y_std < confidence_cutoff)[0].tolist() if confidence_cutoff is not None else []
            idx_confident = idx_query[np.array(idx_confident_)].tolist()
            mask = np.isin(range(len(dataset_pool)), idx + idx_confident)
            idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
            return idx, acquisition, idx_remain
        else:
            return list(range(len(dataset_pool))), [], []

    @property
    def info(self) -> str:
        return (f"PartialQueryClusterExplorativeSelector(batch_size={self.batch_size}, "
                f"cluster_size={self.cluster_size}, query_size={self.query_size})")


class ExploitiveSelector(BaseRandomSelector):
    def __init__(self, target, batch_size: int = 1, seed: int = 0):
        super().__init__(batch_size=batch_size, seed=seed)
        self.target = target

    def __call__(self, model, dataset_pool, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        y_pred = model.predict_value(dataset_pool)
        idx = get_topn_idx(y_pred, n=self.batch_size, target=self.target)

        acquisition = y_pred[np.array(idx)].tolist()

        mask = np.isin(range(len(dataset_pool)), idx)
        idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return f"ExploitiveSelector(batch_size={self.batch_size}, target={self.target})"


class ClusterExploitiveSelector(BaseClusterSelector):
    def __init__(self, target, batch_size: int = 1, cluster_size: int = None, seed: int = 0):
        super().__init__(batch_size=batch_size, cluster_size=cluster_size, seed=seed)
        self.target = target

    def __call__(self, model, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        y_pred = model.predict_value(dataset_pool)
        idx_cluster = get_topn_idx(y_pred, n=self.cluster_size, target=self.target)
        idx = np.array(idx_cluster)[np.array(self.get_idx_cluster(dataset_pool, kernel, idx_cluster))].tolist()

        acquisition = y_pred[np.array(idx)].tolist()

        mask = np.isin(range(len(dataset_pool)), idx)
        idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return (f"ClusterExploitiveSelector(batch_size={self.batch_size}, "
                f"cluster_size={self.cluster_size}, target={self.target})")


class PartialQueryExploitiveSelector(BasePartialQuerySelector, BaseRandomSelector):
    def __init__(self, target, query_size: int, batch_size: int = 1, seed: int = 0):
        super().__init__(batch_size=batch_size, seed=seed)
        self.target = target
        self.query_size = query_size
        assert batch_size <= query_size, "batch_size should be smaller than query_size."

    def __call__(self, model, dataset_pool, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        data_query, idx_query = self.get_partial_data(dataset_pool, self.query_size)
        y_pred = model.predict_value(data_query)
        idx_ = get_topn_idx(y_pred, n=self.batch_size, target=self.target)
        idx = idx_query[np.array(idx_)].tolist()

        acquisition = y_pred[idx_].tolist()

        mask = np.isin(range(len(dataset_pool)), idx)
        idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return f"PartialQueryExploitiveSelector(batch_size={self.batch_size}, query_size={self.query_size}, target={self.target})"


class PartialQueryClusterExploitiveSelector(BaseClusterSelector, BasePartialQuerySelector):
    def __init__(self, target, query_size: int, batch_size: int = 1, cluster_size: int = None, seed: int = 0):
        super().__init__(batch_size=batch_size, cluster_size=cluster_size, seed=seed)
        self.target = target
        self.query_size = query_size
        assert self.cluster_size <= query_size, "cluster_size should be smaller than query_size."
    
    def __call__(self, model, dataset_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        data_query, idx_query = self.get_partial_data(dataset_pool, self.query_size)
        y_pred = model.predict_value(data_query)
        idx_cluster = get_topn_idx(y_pred, n=self.cluster_size, target=self.target)
        idx_ = np.array(idx_cluster)[np.array(self.get_idx_cluster(dataset_pool, kernel, idx_cluster))].tolist()
        idx = np.array(idx_query)[np.array(idx_)].tolist()

        acquisition = y_pred[idx_].tolist()

        mask = np.isin(range(len(dataset_pool)), idx)
        idx_remain = np.arange(len(dataset_pool))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return (f"PartialQueryClusterExploitiveSelector(batch_size={self.batch_size}, "
                f"cluster_size={self.cluster_size}, query_size={self.query_size}, target={self.target})")


class ProbabilityImprovementSelector(BaseRandomSelector):
    pass


class ExpectedImprovementSelector(BaseRandomSelector):
    pass


class UpperConfidenceBoundSelector(BaseRandomSelector):
    pass
