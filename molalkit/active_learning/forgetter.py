#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from molalkit.active_learning.utils import random_choice, get_topn_idx
from molalkit.models.random_forest.RandomForestClassifier import RFClassifier
from molalkit.models.gaussian_process.GaussianProcessRegressor import GPRegressor


class BaseForgetter(ABC):
    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size

    @abstractmethod
    def __call__(self, **kwargs) -> Tuple[List[int], List[float]]:
        pass

    @property
    @abstractmethod
    def info(self) -> str:
        pass


class BaseRandomForgetter(BaseForgetter, ABC):
    """Base Forgetter that uses random seed."""
    def __init__(self, batch_size: int = 1, seed: int = 0):
        super().__init__(batch_size=batch_size)
        np.random.seed(seed)


class RandomForgetter(BaseRandomForgetter):
    def __call__(self, dataset_train, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        idx, idx_remain = random_choice(len(dataset_train), self.batch_size)
        return idx, [], idx_remain

    @property
    def info(self) -> str:
        return f"RandomForgetter(batch_size={self.batch_size})"


class FirstForgetter(BaseForgetter):
    def __call__(self, dataset_train, **kwargs) -> Tuple[List[int], None]:
        return list(range(self.batch_size)), [], list(range(self.batch_size, len(dataset_train)))

    @property
    def info(self) -> str:
        return f"FirstForgetter(batch_size={self.batch_size})"


class MinOOBUncertaintyForgetter(BaseRandomForgetter):
    """ Forget the samples with the lowest out-of-bag (OOB) uncertainty. 
        This method is only applicable to random forest classifier. 
    """
    def __call__(self, model: RFClassifier, dataset_train, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        # assert self.batch_size < len(dataset_train), "batch_size should be less than the size of the dataset."
        assert isinstance(model, RFClassifier), "Only random forest classifier is supported for OOB prediction."
        assert model.oob_score is True, "You should set oob_score=True when training the model."
        y_oob_proba = model.oob_decision_function_
        # uncertainty calculation, normalized into 0 to 1
        y_oob_uncertainty = (0.25 - np.var(y_oob_proba, axis=1)) * 4
        # select the top-n points with least uncertainty
        idx = get_topn_idx(y_oob_uncertainty, n=self.batch_size, target="min")

        acquisition = y_oob_uncertainty[np.array(idx)].tolist()

        mask = np.isin(range(len(dataset_train)), idx)
        idx_remain = np.arange(len(dataset_train))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return f"MinOOBUncertaintyForgetter(batch_size={self.batch_size})"


class MaxOOBUncertaintyForgetter(BaseRandomForgetter):
    """ Forget the samples with the highest out-of-bag (OOB) uncertainty.
        This method is only applicable to random forest classifier.
    """
    def __call__(self, model: RFClassifier, dataset_train, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        # assert self.batch_size < len(dataset_train), "batch_size should be less than the size of the dataset."
        assert isinstance(model, RFClassifier), "Only random forest classifier is supported for OOB prediction."
        assert model.oob_score is True, "You should set oob_score=True when training the model."
        y_oob_proba = model.oob_decision_function_
        # uncertainty calculation, normalized into 0 to 1
        y_oob_uncertainty = (0.25 - np.var(y_oob_proba, axis=1)) * 4
        # select the top-n points with least uncertainty
        idx = get_topn_idx(y_oob_uncertainty, n=self.batch_size, target="max")

        acquisition = y_oob_uncertainty[np.array(idx)].tolist()

        mask = np.isin(range(len(dataset_train)), idx)
        idx_remain = np.arange(len(dataset_train))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return f"MaxOOBUncertaintyForgetter(batch_size={self.batch_size})"


class MinOOBErrorForgetter(BaseRandomForgetter):
    """ Forget the samples with the lowest out-of-bag (OOB) error.
        This method is only applicable to random forest classifier.
    """
    def __call__(self, model: RFClassifier, dataset_train, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        # assert self.batch_size < len(dataset_train), "batch_size should be less than the size of the dataset."
        assert isinstance(model, RFClassifier), "Only random forest classifier is supported for OOB prediction."
        assert model.oob_score is True, "You should set oob_score=True when training the model."
        y_oob_proba = model.oob_decision_function_
        # uncertainty calculation, normalized into 0 to 1
        oob_error = np.absolute(y_oob_proba[:, 1] - dataset_train.y.ravel())
        # select the top-n points with least uncertainty
        idx = get_topn_idx(oob_error, n=self.batch_size, target="min")

        acquisition = oob_error[np.array(idx)].tolist() if idx else []

        mask = np.isin(range(len(dataset_train)), idx)
        idx_remain = np.arange(len(dataset_train))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return f"MinOOBErrorForgetter(batch_size={self.batch_size})"


class MaxOOBErrorForgetter(BaseRandomForgetter):
    """ Forget the samples with the highest out-of-bag (OOB) error.
        This method is only applicable to random forest classifier.
    """
    def __call__(self, model: RFClassifier, dataset_train, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        # assert self.batch_size < len(dataset_train), "batch_size should be less than the size of the dataset."
        assert isinstance(model, RFClassifier), "Only random forest classifier is supported for OOB prediction."
        assert model.oob_score is True, "You should set oob_score=True when training the model."
        y_oob_proba = model.oob_decision_function_
        # uncertainty calculation, normalized into 0 to 1
        oob_error = np.absolute(y_oob_proba[:, 1] - dataset_train.y.ravel())
        # select the top-n points with least uncertainty
        idx = get_topn_idx(oob_error, n=self.batch_size, target="max")

        acquisition = oob_error[np.array(idx)].tolist() if idx else []

        mask = np.isin(range(len(dataset_train)), idx)
        idx_remain = np.arange(len(dataset_train))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return f"MaxOOBErrorForgetter(batch_size={self.batch_size})"


class MinLOOUncertaintyForgetter(BaseRandomForgetter):
    def __call__(self, model: GPRegressor, dataset_train, **kwargs) -> Tuple[List[int], List[float]]:
        """ Forget the samples with the lowest Leave-one-out cross-validation (LOOCV) uncertainty.
        Parameters
        ----------
        model: Only Gaussian process regressor is supported due to efficient LOOCV of GPR.
        dataset_train: The dataset to forget.

        Returns
        -------
        The index and the acquisition value of samples to forget.
        """
        # assert self.batch_size < len(dataset_train), "batch_size should be less than the size of the dataset."
        assert isinstance(model, GPRegressor), "Only Gaussian process regressor is supported for LOOCV prediction."
        y_loocv, y_loocv_std = model.predict_loocv(dataset_train.X, dataset_train.y.ravel(), return_std=True)
        # uncertainty calculation, normalized into 0 to 1
        loo_uncertainty = y_loocv_std
        # select the top-n points with least uncertainty
        idx = get_topn_idx(loo_uncertainty, n=self.batch_size, target="min")

        acquisition = loo_uncertainty[np.array(idx)].tolist() if idx else []

        mask = np.isin(range(len(dataset_train)), idx)
        idx_remain = np.arange(len(dataset_train))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return f"MinLOOUncertaintyForgetter(batch_size={self.batch_size})"


class MaxLOOUncertaintyForgetter(BaseRandomForgetter):
    def __call__(self, model: GPRegressor, dataset_train, **kwargs) -> Tuple[List[int], List[float]]:
        """ Forget the samples with the highest Leave-one-out cross-validation (LOOCV) uncertainty.
        Parameters
        ----------
        model: Only Gaussian process regressor is supported due to efficient LOOCV of GPR.
        dataset_train: The dataset to forget.

        Returns
        -------
        The index and the acquisition value of samples to forget.
        """
        # assert self.batch_size < len(dataset_train), "batch_size should be less than the size of the dataset."
        assert isinstance(model, GPRegressor), "Only Gaussian process regressor is supported for LOOCV prediction."
        y_loocv, y_loocv_std = model.predict_loocv(dataset_train.X, dataset_train.y.ravel(), return_std=True)
        # uncertainty calculation, normalized into 0 to 1
        loo_uncertainty = y_loocv_std
        # select the top-n points with least uncertainty
        idx = get_topn_idx(loo_uncertainty, n=self.batch_size, target="max")

        acquisition = loo_uncertainty[np.array(idx)].tolist() if idx else []

        mask = np.isin(range(len(dataset_train)), idx)
        idx_remain = np.arange(len(dataset_train))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return f"MaxLOOUncertaintyForgetter(batch_size={self.batch_size})"


class MinLOOErrorForgetter(BaseRandomForgetter):
    def __call__(self, model: GPRegressor, dataset_train, **kwargs) -> Tuple[List[int], List[float]]:
        """ Forget the samples with the lowest Leave-one-out cross-validation (LOOCV) error.
        Parameters
        ----------
        model: Only Gaussian process regressor is supported due to efficient LOOCV of GPR.
        dataset_train: The dataset to forget.

        Returns
        -------
        The index and the acquisition value of samples to forget.
        """
        # assert self.batch_size < len(dataset_train), "batch_size should be less than the size of the dataset."
        assert isinstance(model, GPRegressor), "Only Gaussian process regressor is supported for LOOCV prediction."
        y_loocv = model.predict_loocv(dataset_train.X, dataset_train.y.ravel(), return_std=False)
        # uncertainty calculation, normalized into 0 to 1
        loo_error = np.absolute(y_loocv - dataset_train.y.ravel())
        # select the top-n points with least uncertainty
        idx = get_topn_idx(loo_error, n=self.batch_size, target="min")

        acquisition = loo_error[np.array(idx)].tolist() if idx else []

        mask = np.isin(range(len(dataset_train)), idx)
        idx_remain = np.arange(len(dataset_train))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return f"MinOOBErrorForgetter(batch_size={self.batch_size})"


class MaxLOOErrorForgetter(BaseRandomForgetter):
    def __call__(self, model: GPRegressor, dataset_train, **kwargs) -> Tuple[List[int], List[float]]:
        """ Forget the samples with the highest Leave-one-out cross-validation (LOOCV) error.
        Parameters
        ----------
        model: Only Gaussian process regressor is supported due to efficient LOOCV of GPR.
        dataset_train: The dataset to forget.

        Returns
        -------
        The index and the acquisition value of samples to forget.
        """
        # assert self.batch_size < len(dataset_train), "batch_size should be less than the size of the dataset."
        assert isinstance(model, GPRegressor), "Only Gaussian process regressor is supported for LOOCV prediction."
        y_loocv = model.predict_loocv(dataset_train.X, dataset_train.y.ravel(), return_std=False)
        # uncertainty calculation, normalized into 0 to 1
        loo_error = np.absolute(y_loocv - dataset_train.y.ravel())
        # select the top-n points with least uncertainty
        idx = get_topn_idx(loo_error, n=self.batch_size, target="max")

        acquisition = loo_error[np.array(idx)].tolist() if idx else []

        mask = np.isin(range(len(dataset_train)), idx)
        idx_remain = np.arange(len(dataset_train))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return f"MaxLOOErrorForgetter(batch_size={self.batch_size})"
