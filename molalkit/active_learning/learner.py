#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, List, Callable
import os
import json
import pickle
import pandas as pd
import numpy as np
from mgktools.evaluators.metric import Metric
from molalkit.active_learning.selector import BaseSelector, RandomSelector
from molalkit.active_learning.forgetter import BaseForgetter, RandomForgetter, FirstForgetter
from molalkit.active_learning.utils import eval_metric_func
from molalkit.data.utils import get_subset_from_uidx


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ActiveLearningResult:
    def __init__(self, n_iter: int):
        self.n_iter = n_iter
        self.uidx_before = []
        self.uidx_select = []
        self.acquisition_select = []
        self.uidx_forget = []
        self.acquisition_forget = []
        self.uidx_after = []
        self.performance = dict()
        # Generic loss statistics (always available when training occurs)
        self.loss_stats = dict()


class ActiveLearningTrajectory:
    def __init__(self, metrics: List[str] = None):
        self.metrics = metrics
        self.results = []

    def update_performance(self, column_name: str, results_dict: Dict) -> bool:
        results = [alr.performance.get(column_name) for alr in self.results]
        if results[0] is not None:
            results_dict[column_name] = results
            return True
        else:
            return False

    def get_results(self) -> Dict:
        results_dict = dict()
        results_dict["n_iter"] = [alr.n_iter for alr in self.results]
        # collect the results of various metrics during active learning
        if self.metrics is not None:
            for metric in self.metrics:
                for i in range(100):
                    tag = self.update_performance(f"{metric}-model_{i}", results_dict)
                    if not tag:
                        break
        # collect the results of top data selected during active learning
        self.update_performance("top_score", results_dict)
        results_dict["uidx_before"] = [json.dumps(alr.uidx_before, cls=NpEncoder) for alr in self.results]
        results_dict["uidx_select"] = [json.dumps(alr.uidx_select, cls=NpEncoder) for alr in self.results]
        results_dict["acquisition_select"] = [json.dumps(alr.acquisition_select, cls=NpEncoder) for alr in self.results]
        results_dict["uidx_forget"] = [json.dumps(alr.uidx_forget, cls=NpEncoder) for alr in self.results]
        results_dict["acquisition_forget"] = [json.dumps(alr.acquisition_forget, cls=NpEncoder) for alr in self.results]
        results_dict["uidx_after"] = [json.dumps(alr.uidx_after, cls=NpEncoder) for alr in self.results]
        return results_dict


class ActiveLearner:
    def __init__(self, save_dir: str, selector: BaseSelector, forgetter: BaseForgetter,
                 models, id2datapoints: List[Dict],
                 datasets_train, datasets_pool, datasets_val = None,
                 metrics: List[Metric] = None, top_uidx: List[int] = None,
                 kernel: Callable = None, detail: bool = False):
        """ Active learning class.
        
        Parameters
        save_dir: str
            Directory to save the results of active learning.
        selector: BaseSelector
            Selection method used to select the data from the pool set.
        forgetter: BaseForgetter
            Forget method used to forget the data from the training set.
        models:
            ML models used in active learning. [selector, evaluator1, evaluator2, ...]
        datasets_train:
            Training datasets used to train the model. [selector, evaluator1, evaluator2, ...]
        dataset_pool:
            Pool set used to select the data during active learning.
        dataset_val:
            Validation set used to evaluate the model performance.
        metrics: List[Metric] = None
            Metrics used to evaluate the ML model performance. This is usually used in explorative 
            active learning in which the goal is to build a model with the best performance.
        top_uidx: List[int] = None
            List of data index that are considered as top data. If this is given, the percentage of 
            top data selected by active learning will be outputed. This is a metric that usually used 
            in exploitive active learning.
        kernel: Callable = None
            Kernel function used in cluster selection method.
        detail: bool = False
            If True, the detailed information of the model performance at each iteration will be outputed.
        """
        self.save_dir = save_dir
        self.selector = selector
        self.forgetter = forgetter
        self.models = models
        self.id2datapoints = id2datapoints
        self.datasets_train = datasets_train
        self.datasets_pool = datasets_pool
        self.datasets_val = datasets_val
        # input for model performance evaluation
        self.metrics = metrics
        self.top_uidx = top_uidx
        # kernel used for cluster selection method
        self.kernel = kernel
        self.detail = detail
        # initialize the active learning trajectory
        self.current_iter = 0
        self.model_fitted = False
        self.active_learning_traj = ActiveLearningTrajectory(metrics=self.metrics)

    @property
    def train_size(self) -> int:
        return len(self.datasets_train[0])

    @property
    def val_size(self) -> int:
        return 0 if self.datasets_val is None else len(self.datasets_val[0])

    def step_select(self, stop_cutoff: float = None, confidence_cutoff: float = None):
        alr = ActiveLearningResult(self.current_iter)
        alr.uidx_before = [data.uidx for data in self.datasets_train[0]]
        assert self.selector is not None, "You need to provide a selector before step_select()."
        # train the model if it is not trained in the evaluation step, and the selection method is not random.
        if not self.model_fitted and not isinstance(self.selector, RandomSelector):
            self.models[0].fit_molalkit(self.datasets_train[0], iteration=self.current_iter)
        selected_idx, acquisition, remain_idx = self.selector(model=self.models[0],
                                                              dataset_pool=self.datasets_pool[0],
                                                              kernel=self.kernel,
                                                              stop_cutoff=stop_cutoff,
                                                              confidence_cutoff=confidence_cutoff)
        alr.uidx_select = [self.datasets_pool[0].data[idx].uidx for idx in selected_idx]
        alr.acquisition_select = acquisition
        # transfer data from pool to train.
        alr.uidx_after = alr.uidx_before + alr.uidx_select
        uidx_pool = [self.datasets_pool[0].data[idx].uidx for idx in remain_idx]
        ### update the dataset_train and dataset_pool
        for i, id2datapoint in enumerate(self.id2datapoints):
            self.datasets_train[i] = get_subset_from_uidx(self.datasets_train[i], id2datapoint, alr.uidx_after)
            self.datasets_pool[i] = get_subset_from_uidx(self.datasets_pool[i], id2datapoint, uidx_pool)
        # Collect per-iteration loss statistics immediately after (re)training (selector only)
        try:
            if hasattr(self.models[0], 'get_cbp_stats'):
                stats = self.models[0].get_cbp_stats()
                if stats:
                    alr.loss_stats = stats
        except Exception:
            pass
        # set the model unfitted because new data is added.
        self.model_fitted = False
        self.current_iter += 1
        self.active_learning_traj.results.append(alr)

    def step_forget(self, stop_cutoff: float = None, confidence_cutoff: float = None):
        alr = ActiveLearningResult(self.current_iter)
        alr.uidx_before = [data.uidx for data in self.datasets_train[0]]
        assert self.forgetter is not None, "You need to provide a forgetter before step_forget()."
        # train the model if the forgetter is not random or first.
        if not self.model_fitted and not self.forgetter.__class__ in [RandomForgetter, FirstForgetter]:
            self.models[0].fit_molalkit(self.datasets_train[0], iteration=self.current_iter)
        # forget algorithm is applied.
        forget_idx, acquisition, remain_idx = self.forgetter(model=self.models[0],
                                                             dataset_train=self.datasets_train[0],
                                                             stop_cutoff=stop_cutoff,
                                                             confidence_cutoff=confidence_cutoff)
        alr.uidx_forget = [self.datasets_train[0].data[i].uidx for i in forget_idx]
        alr.acquisition_forget = acquisition
        # transfer data from train to pool.
        alr.uidx_after = [self.datasets_train[0].data[i].uidx for i in remain_idx]# alr.uidx_before + alr.uidx_select
        uidx_pool = [data.uidx for data in self.datasets_pool[0]] + alr.uidx_forget
        ### update the dataset_train and dataset_pool
        for i, id2datapoint in enumerate(self.id2datapoints):
            self.datasets_train[i] = get_subset_from_uidx(self.datasets_train[i], id2datapoint, alr.uidx_after)
            self.datasets_pool[i] = get_subset_from_uidx(self.datasets_pool[i], id2datapoint, uidx_pool)
        # Collect per-iteration loss statistics after (re)training (selector only)
        try:
            if hasattr(self.models[0], 'get_cbp_stats'):
                stats = self.models[0].get_cbp_stats()
                if stats:
                    alr.loss_stats = stats
        except Exception:
            pass
        # set the model unfitted because new data is added.
        self.model_fitted = False
        self.current_iter += 1
        self.active_learning_traj.results.append(alr)

    def evaluate(self):
        if len(self.active_learning_traj.results) == 0:
            alr = ActiveLearningResult(self.current_iter - 1)
            alr.uidx_before = []
            alr.uidx_after = [data.uidx for data in self.datasets_train[0]]
            self.active_learning_traj.results.append(alr)
        else:
            alr = self.active_learning_traj.results[-1]
        # evaluate the prediction performance of ML model on the validation set
        if self.metrics is not None:
            for i, model in enumerate(self.models):
                model.fit_molalkit(self.datasets_train[i])
                y_pred = model.predict_value(self.datasets_val[i])
                y_pred = [data.targets[0] if data.uidx in alr.uidx_after else y_pred[i] for i, data in enumerate(self.datasets_val[i])]
                if self.detail:
                    df = pd.DataFrame({"true": self.datasets_val[i].y.ravel(), "pred": y_pred})
                    df.to_csv(os.path.join(self.save_dir, f"model_{i}-iter_{self.current_iter}.csv"), index=False)
                for metric in self.metrics:
                    metric_value = eval_metric_func(self.datasets_val[i].y.ravel(), y_pred, metric=metric)
                    alr.performance[f"{metric}-model_{i}"] = metric_value
                # Collect loss statistics from selector (available regardless of CBP)
                if i == 0 and hasattr(model, 'get_cbp_stats'):
                    stats = model.get_cbp_stats()
                    if stats:
                        alr.loss_stats = stats
                        if 'final_loss' in stats:
                            print(f'Loss Stats - Final loss: {stats.get("final_loss", "N/A")}')
            self.model_fitted = True

        # Always try to collect loss stats for selector, even when metrics is None
        try:
            model = self.models[0]
            if hasattr(model, 'get_cbp_stats'):
                stats = model.get_cbp_stats()
                if stats:
                    alr.loss_stats = stats
        except Exception:
            pass
        # evaluate the percentage of top data selected in the training set
        if self.top_uidx is not None:
            alr.performance["top_score"] = self.get_top_score(self.datasets_train[0], self.top_uidx)

    def write_traj(self):
        df_traj = pd.DataFrame(self.active_learning_traj.get_results())
        df_traj.to_csv(os.path.join(self.save_dir, "al_traj.csv"), index=False)
        # Write a lightweight iteration summary for quick reading/plotting
        try:
            # Collect final_loss per AL iteration from loss_stats if available
            final_losses = []
            for alr in self.active_learning_traj.results:
                stats = getattr(alr, 'loss_stats', None)
                final_losses.append(stats.get('final_loss') if isinstance(stats, dict) else None)
            lite = pd.DataFrame({
                "n_iter": df_traj.get("n_iter", []),
                # train_size = len(uidx_after) per iteration
                "train_size": df_traj.get("uidx_after", []).apply(lambda s: len(json.loads(s)) if isinstance(s, str) else None),
                "selected_count": df_traj.get("uidx_select", []).apply(lambda s: len(json.loads(s)) if isinstance(s, str) else 0),
                "forgot_count": df_traj.get("uidx_forget", []).apply(lambda s: len(json.loads(s)) if isinstance(s, str) else 0),
                # Optional columns may not exist depending on task
                "top_score": df_traj.get("top_score", None),
                # Final training loss of this AL iteration (last epoch's average loss)
                "final_loss": final_losses,
            })
            lite.to_csv(os.path.join(self.save_dir, "al_traj_light.csv"), index=False)
        except Exception:
            pass
        
        # Write aggregated loss statistics
        self.write_loss_stats()
    
    def write_loss_stats(self):
        """Write aggregated loss statistics to a separate JSON file (always)."""
        try:
            stats_list = []
            for alr in self.active_learning_traj.results:
                stats = getattr(alr, 'loss_stats', None)
                if stats:
                    stats_list.append({
                        'iteration': alr.n_iter,
                        **stats
                    })

            path = os.path.join(self.save_dir, 'loss_stats.json')
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        old = json.load(f)
                except Exception:
                    old = []
            else:
                old = []
            # Merge by iteration (last write wins)
            merged = {d['iteration']: d for d in old if isinstance(d, dict) and 'iteration' in d}
            for d in stats_list:
                merged[d['iteration']] = d
            merged_list = [merged[k] for k in sorted(merged.keys())]
            if merged_list:
                with open(path, 'w') as f:
                    json.dump(merged_list, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def get_top_score(dataset, top_uidx) -> float:
        N_top_k = 0
        for data in dataset:
            if data.uidx in top_uidx:
                N_top_k += 1
        return N_top_k / len(top_uidx)

    def save(self, path, filename="al.pkl", overwrite=False):
        f_al = os.path.join(path, filename)
        if os.path.isfile(f_al) and not overwrite:
            raise RuntimeError(
                f"Path {f_al} already exists. To overwrite, set "
                "`overwrite=True`."
            )
        store = self.__dict__.copy()
        # Temporarily remove unpicklable attributes, then restore after pickling.
        saved_states = []
        for model in store["models"]:
            state = {}
            # Chemprop TrainArgs/PredictArgs are unpicklable, transform into dict.
            if hasattr(model, "chemprop_train_args"):
                model.chemprop_train_args = model.chemprop_train_args.as_dict()
                model.chemprop_predict_args = model.chemprop_predict_args.as_dict()
            # Logger contains FileHandler with TextIOWrapper (unpicklable).
            state['logger'] = getattr(model, 'logger', None)
            model.logger = None
            # CBP trainer has unpicklable attributes.
            cbp_trainer = getattr(model, 'cbp_trainer', None)
            if cbp_trainer is not None:
                # cbp_trainer.args is the same TrainArgs reference (unpicklable).
                state['cbp_args'] = getattr(cbp_trainer, 'args', None)
                cbp_trainer.args = None
                # cbp_logger.log_file is a TextIOWrapper (unpicklable).
                cbp_logger = getattr(cbp_trainer, 'cbp_logger', None)
                if cbp_logger is not None:
                    state['cbp_log_file'] = getattr(cbp_logger, 'log_file', None)
                    cbp_logger.log_file = None
            saved_states.append(state)
        pickle.dump(store, open(f_al, "wb"), protocol=4)
        # Restore all attributes after pickling.
        for model, state in zip(store["models"], saved_states):
            if hasattr(model, "chemprop_train_args"):
                from chemprop.args import TrainArgs, PredictArgs
                model.chemprop_train_args = TrainArgs().from_dict(model.chemprop_train_args, skip_unsettable=True)
                model.chemprop_predict_args = PredictArgs().from_dict(model.chemprop_predict_args, skip_unsettable=True)
            model.logger = state.get('logger')
            cbp_trainer = getattr(model, 'cbp_trainer', None)
            if cbp_trainer is not None:
                if 'cbp_args' in state:
                    cbp_trainer.args = state['cbp_args']
                cbp_logger = getattr(cbp_trainer, 'cbp_logger', None)
                if cbp_logger is not None and 'cbp_log_file' in state:
                    cbp_logger.log_file = state['cbp_log_file']

    @classmethod
    def load(cls, path, filename="al.pkl"):
        f_al = os.path.join(path, filename)
        store = pickle.load(open(f_al, "rb"))
        # Restore unpicklable attributes.
        for model in store["models"]:
            # Transform Chemprop TrainArgs/PredictArgs from dict back to objects.
            if hasattr(model, "chemprop_train_args"):
                from chemprop.args import TrainArgs, PredictArgs
                model.chemprop_train_args = TrainArgs().from_dict(model.chemprop_train_args, skip_unsettable=True)
                model.chemprop_predict_args = PredictArgs().from_dict(model.chemprop_predict_args, skip_unsettable=True)
            # Restore cbp_trainer references that were nulled during save.
            cbp_trainer = getattr(model, 'cbp_trainer', None)
            if cbp_trainer is not None:
                # Restore args reference to the reconstructed TrainArgs.
                if hasattr(model, 'chemprop_train_args'):
                    cbp_trainer.args = model.chemprop_train_args
                # Reopen cbp_logger log file.
                cbp_logger = getattr(cbp_trainer, 'cbp_logger', None)
                if cbp_logger is not None and hasattr(cbp_logger, '_open_log_file'):
                    try:
                        cbp_logger._open_log_file()
                    except Exception:
                        cbp_logger.log_file = None
        input = {}
        for key in ["save_dir", "selector", "forgetter", "models",
                    "id2datapoints", "datasets_train", "datasets_pool"]:
            input[key] = store[key]
        learner = cls(**input)
        learner.__dict__.update(**store)
        return learner
