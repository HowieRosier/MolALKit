#!/usr/bin/env python
# -*- coding: utf-8 -*-
from xgboost import XGBRegressor as XGBR
from molalkit.models.base import BaseSklearnModel


class XGBRegressor(XGBR, BaseSklearnModel):
    def fit_molalkit(self, train_data, iteration: int = 0):
        return self.fit_molalkit_(train_data, self)

    def predict_uncertainty(self, pred_data):
        X = pred_data.X
        # TODO
        raise NotImplementedError

    def predict_value(self, pred_data):
        X = pred_data.X
        return super().predict(X)
