#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from graphdot.model.gaussian_process import GaussianProcessRegressor as GPR
from molalkit.models.base import BaseModel


class GPRegressor(GPR, BaseModel):
    def __init__(self, *args, **kwargs):
        uncertainty_type = kwargs.pop("uncertainty_type")
        super().__init__(*args, **kwargs)
        self.uncertainty_type = uncertainty_type

    def fit_molalkit(self, train_data, **kwargs):
        X = train_data.X
        y = train_data.y
        super().fit(X, y, **kwargs)

    def predict_uncertainty(self, pred_data):
        X = pred_data.X
        if self.uncertainty_type == "value":
            preds = self.predict(X).reshape(-1, 1)
            preds = np.concatenate([preds, 1 - preds], axis=1)
            return (0.25 - np.var(preds, axis=1)) * 4
        elif self.uncertainty_type == "uncertainty":
            return self.predict(X, return_std=True)[1]

    def predict_value(self, pred_data):
        X = pred_data.X
        return self.predict(X)
