#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import check_is_fitted, _partition_estimators, Parallel, delayed, _accumulate_prediction
from molalkit.models.base import BaseSklearnModel


class RFRegressor(RandomForestRegressor, BaseSklearnModel):
    def fit_molalkit(self, train_data, iteration: int = 0):
        return self.fit_molalkit_(train_data, self)

    def predict_uncertainty(self, pred_data):
        X = pred_data.X
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)
        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        results = Parallel(
            n_jobs=n_jobs, verbose=self.verbose, prefer="processes")(
            delayed(e.predict)(X)
            for e in self.estimators_)
        return np.asarray(results).std(axis=0)

    def predict_value(self, pred_data):
        X = pred_data.X
        return super().predict(X)
