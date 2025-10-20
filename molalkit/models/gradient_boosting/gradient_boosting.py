#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import GradientBoostingRegressor as GBR
from molalkit.models.base import BaseSklearnModel


class GradientBoostingClassifier(GBC, BaseSklearnModel):
    def fit_molalkit(self, train_data, iteration: int = 0):
        return self.fit_molalkit_(train_data, self)

    def predict_uncertainty(self, pred_data):
        return self.predict_uncertainty_c(pred_data, self)

    def predict_value(self, pred_data):
        return self.predict_value_c(pred_data, self)


class GradientBoostingRegressor(GBR, BaseSklearnModel):
    def fit_molalkit(self, train_data, iteration: int = 0):
        return self.fit_molalkit_(train_data, self)

    def predict_uncertainty(self, pred_data):
        return self.predict_uncertainty_r(pred_data, self)

    def predict_value(self, pred_data):
        return self.predict_value_r(pred_data, self)
