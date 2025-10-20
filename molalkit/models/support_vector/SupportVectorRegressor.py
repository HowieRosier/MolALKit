#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.svm import SVR
from molalkit.models.base import BaseSklearnModel


class SVRegressor(SVR, BaseSklearnModel):
    def fit_molalkit(self, train_data, iteration: int = 0):
        return self.fit_molalkit_(train_data, self)

    def predict_uncertainty(self, pred_data):
        raise ValueError("SVRegressor does not support uncertainty prediction")

    def predict_value(self, pred_data):
        X = pred_data.X
        return super().predict(X)
