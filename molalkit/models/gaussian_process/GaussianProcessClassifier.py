#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mgktools.models.classification.gpc.gpc import GPC_MGK
from molalkit.models.base import BaseSklearnModel


class GPClassifier(GPC_MGK, BaseSklearnModel):
    def fit_molalkit(self, train_data, iteration: int = 0):
        return self.fit_molalkit_(train_data, self)

    def predict_uncertainty(self, pred_data):
        return self.predict_uncertainty_c(pred_data, self)

    def predict_value(self, pred_data):
        return self.predict_value_c(pred_data, self)
