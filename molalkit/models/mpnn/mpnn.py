#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
from typing import List, Literal
from tqdm import trange
from logging import Logger
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from chemprop.data import get_class_sizes, MoleculeDataLoader, get_task_names
from chemprop.utils import build_optimizer, build_lr_scheduler, makedirs, load_mpn_model, save_checkpoint, load_checkpoint
from chemprop.nn_utils import param_count, param_count_all
from chemprop.models import MoleculeModel
from chemprop.train.loss_functions import get_loss_func
from chemprop.train import train
from chemprop.models.cbp_trainer import ContinualBackpropTrainer
from chemprop.args import TrainArgs, PredictArgs
from chemprop.train.make_predictions import set_features, predict_and_save
from molalkit.data.utils import get_subset_from_idx


class MPNN:
    def __init__(self,
                 # TrainArgs parameters
                 save_dir: str, data_path: str,
                 dataset_type: Literal["regression", "classification", "multiclass", "spectra"],
                 loss_function: Literal["mse", "bounded_mse", "binary_cross_entropy", "cross_entropy", "mcc", "sid",
                                        "wasserstein", "mve", "evidential", "dirichlet"],
                 smiles_columns: List[str] = None, target_columns: List[str] = None,
                 multiclass_num_classes: int = 3,
                 features_generator=None,
                 no_features_scaling: bool = False,
                 features_only: bool = False,
                 features_size: int = 0,
                 epochs: int = 30,
                 depth: int = 3,
                 hidden_size: int = 300,
                 ffn_num_layers: int = 2,
                 ffn_hidden_size: int = None,
                 dropout: float = 0.0,
                 batch_size: int = 50,
                 ensemble_size: int = 1,
                 number_of_molecules: int = 1,
                 mpn_shared: bool = False,
                 atom_messages: bool = False,
                 undirected: bool = False,
                 n_jobs: int = 8,
                 class_balance: bool = False,
                 checkpoint_dir: str = None,
                 checkpoint_frzn: str = None,
                 frzn_ffn_layers: int = 0,
                 freeze_first_only: bool = False,
                 mpn_path: str = None,
                 freeze_mpn: bool = False,
                 seed: int = 0,
                 # PredictArgs parameters
                 uncertainty_method: Literal["mve", "ensemble", "evidential_epistemic", "evidential_aleatoric",
                                             "evidential_total", "classification", "dropout", "spectra_roundrobin"] = None,
                 uncertainty_dropout_p: float = 0.1,
                 dropout_sampling_size: int = 10,
                 # other parameters
                 continuous_fit: bool = False,
                 # L2 (weight decay)
                 weight_decay: float = 0.0,
                 perturb_sigma: float = 0.0,
                 # CBP (Continual Backpropagation) parameters
                 cbp: bool = False,
                 maturity_threshold: int = 100,
                 replacement_rate: float = 0.001,
                 decay_rate: float = 0.95,
                 util_type: str = 'contribution',
                 enable_gradient_logging: bool = True,
                 gradient_log_frequency: int = 1000000,  # Default: epoch-level logging
                 logger: Logger = None,
                 # logging controls
                 log_iter_loss: bool = False,
                 ):
        args = TrainArgs()
        args.save_dir = save_dir
        args.data_path = data_path
        args.dataset_type = dataset_type
        args.loss_function = loss_function
        args.smiles_columns = smiles_columns
        args.target_columns = target_columns
        args.multiclass_num_classes = multiclass_num_classes
        args.features_generator = features_generator
        args.no_features_scaling = no_features_scaling
        args.features_only = features_only
        args.features_size = features_size
        args.epochs = epochs
        args.depth = depth
        args.hidden_size = hidden_size
        args.ffn_num_layers = ffn_num_layers
        args.ffn_hidden_size = ffn_hidden_size
        args.dropout = dropout
        args.batch_size = batch_size
        args.ensemble_size = ensemble_size
        args.number_of_molecules = number_of_molecules
        args.mpn_shared = mpn_shared
        args.atom_messages = atom_messages
        args.undirected = undirected
        args.num_workers = n_jobs
        args.class_balance = class_balance
        args.checkpoint_dir = checkpoint_dir
        args.checkpoint_frzn = checkpoint_frzn
        args.frzn_ffn_layers = frzn_ffn_layers
        args.freeze_first_only = freeze_first_only
        args.mpn_path = mpn_path
        args.freeze_mpn = freeze_mpn
        args.seed = seed
        # L2 regularization
        args.weight_decay = weight_decay
        # Set CBP parameters
        args.cbp = cbp
        args.maturity_threshold = maturity_threshold
        args.replacement_rate = replacement_rate
        args.decay_rate = decay_rate
        args.util_type = util_type
        args.enable_gradient_logging = enable_gradient_logging
        args.gradient_log_frequency = gradient_log_frequency
        args.process_args()
        args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                         target_columns=args.target_columns, ignore_columns=args.ignore_columns)
        args._parsed = True
        self.chemprop_train_args = args
        self.continuous_fit = continuous_fit
        self.logger = logger
        # SnP: perturb noise std; shrink comes from weight_decay
        self.perturb_sigma = perturb_sigma
        self.cbp_stats = {}
        self.cbp_trainer = None  # Single CBP trainer persists across iterations
        if log_iter_loss:
            try:
                # Log every batch inside chemprop train loop
                args.log_frequency = 1
            except Exception:
                pass
        args_predict = PredictArgs()
        args_predict.uncertainty_method = uncertainty_method
        args_predict.uncertainty_dropout_p = uncertainty_dropout_p
        args_predict.dropout_sampling_size = dropout_sampling_size
        args_predict.test_path = "fake"
        args_predict.preds_path = "fake"
        args_predict._parsed = True
        args_predict.checkpoint_paths = [None] * args.ensemble_size
        self.chemprop_predict_args = args_predict

    def fit_molalkit(self, train_data, iteration: int = 0):
        if not self.continuous_fit and torch.cuda.is_available():
            torch.cuda.empty_cache()
        args = self.chemprop_train_args
        args.train_data_size = len(train_data)
        logger = self.logger

        if logger is not None:
            debug, info = logger.debug, logger.info
        else:
            debug = info = print

        # Set pytorch seed for random initial weights
        torch.manual_seed(args.pytorch_seed)

        if args.dataset_type == "classification":
            train_class_sizes = get_class_sizes(train_data, proportion=False)
            args.train_class_sizes = train_class_sizes

        if args.features_scaling:
            features_scaler = train_data.normalize_features(
                replace_nan_token=0)
        else:
            features_scaler = None

        # Scale training targets (regression only)
        if args.dataset_type == "regression":
            debug("Fitting scaler")
            scaler = train_data.normalize_targets()
            args.spectra_phase_mask = None
        else:
            args.spectra_phase_mask = None
            scaler = None

        # Get loss function
        loss_func = get_loss_func(args)

        train_data_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            class_balance=args.class_balance,
            shuffle=True,
            seed=args.seed
        )

        if args.class_balance:
            debug(
                f"With class_balance, effective train size = {train_data_loader.iter_size:,}")

        if self.continuous_fit and hasattr(self, "models"):
            assert len(self.models) == args.ensemble_size
            # Ensure cbp_trainers list exists when continuing
            if args.cbp and not hasattr(self, "cbp_trainers"):
                self.cbp_trainers = []
        else:
            self.models = []
            # Reset CBP trainers when models are reset
            self.cbp_trainers = []

        self.scalers = []
        for model_idx in range(args.ensemble_size):
            save_dir = os.path.join(args.save_dir, f"model_{model_idx}")
            makedirs(save_dir)
            writer = None
            if self.continuous_fit and len(self.models) == args.ensemble_size:
                debug(f"Loading model {model_idx} from previous iteration")
                model = self.models[model_idx]
            else:
                debug(f"Building model {model_idx} from scratch")
                model = MoleculeModel(args)
                if args.cuda:
                    debug("Moving model to cuda")
                model = model.to(args.device)

            if args.mpn_path is not None:
                debug(f"Loading MPN parameters from {args.mpn_path}.")
                model = load_mpn_model(
                    model=model, path=args.mpn_path, current_args=args, logger=logger)

            debug(model)

            if args.freeze_mpn:
                debug(f"Number of unfrozen parameters = {param_count(model):,}")
                debug(f"Total number of parameters = {param_count_all(model):,}")
            else:
                debug(f"Number of parameters = {param_count_all(model):,}")

            if args.cbp:
                debug(f"CBP enabled - maturity: {args.maturity_threshold}, rate: {args.replacement_rate}, decay: {args.decay_rate}, util: {args.util_type}")

            cbp_trainer = None
            cbp_log_dir = None

            if args.cbp:
                cbp_log_dir = os.path.join(save_dir, 'cbp_logs')

                if self.cbp_trainer is None:
                    makedirs(cbp_log_dir)
                    enable_gradient_logging = getattr(args, 'enable_gradient_logging', True)
                    gradient_log_frequency = getattr(args, 'gradient_log_frequency', 1000000)

                    self.cbp_trainer = ContinualBackpropTrainer(
                        model=model,
                        args=args,
                        step_size=args.init_lr,
                        replacement_rate=args.replacement_rate,
                        decay_rate=args.decay_rate,
                        maturity_threshold=args.maturity_threshold,
                        util_type=args.util_type,
                        accumulate=getattr(args, 'accumulate', True),
                        enable_cbp_logging=True,
                        log_dir=cbp_log_dir,
                        enable_gradient_logging=enable_gradient_logging,
                        gradient_log_frequency=gradient_log_frequency
                    )

                    # Restore optimizer state if loading from checkpoint
                    if hasattr(self, '_pending_cbp_state') and model_idx in self._pending_cbp_state:
                        cbp_state = self._pending_cbp_state[model_idx]
                        self.cbp_trainer.optimizer.load_state_dict(cbp_state['optimizer_state_dict'])
                        debug(f"Restored CBP optimizer state from checkpoint (iteration {cbp_state.get('iteration', 0)})")
                        del self._pending_cbp_state[model_idx]

                    debug(f"CBP trainer initialized: {cbp_log_dir}")
                    if enable_gradient_logging:
                        debug(f"  Gradient logging: epoch-level" if gradient_log_frequency >= 1000000 else f"  Gradient logging: batch-level")
                else:
                    self.cbp_trainer.model = model
                    debug(f"Reusing CBP trainer across iterations")
                    if self.cbp_trainer.cbp_logger:
                        self.cbp_trainer.cbp_logger.mark_iteration_start(iteration)

                cbp_trainer = self.cbp_trainer
                optimizer = cbp_trainer.optimizer
                debug(f"Using CBP trainer's optimizer")
            else:
                # Non-CBP mode: reuse optimizer for continuous learning
                if self.continuous_fit and hasattr(self, '_optimizer') and self._optimizer is not None:
                    optimizer = self._optimizer
                    debug(f"Reusing optimizer across iterations")
                else:
                    optimizer = build_optimizer(model, args)
                    self._optimizer = optimizer
                    debug(f"Using standard optimizer")
            
            # Learning rate scheduler [enabled by default]
            # scheduler = build_lr_scheduler(optimizer, args)
            
            # ===== Alternative Schedulers (comment/uncomment to switch) =====
            # 1. DummyLRScheduler: Keep LR constant throughout training
            class DummyLRScheduler:
                def __init__(self, optimizer):
                    self.optimizer = optimizer
                def get_lr(self):
                    return [group['lr'] for group in self.optimizer.param_groups]
                def step(self, *args, **kwargs):
                    return
            scheduler = DummyLRScheduler(optimizer)
            
            # 2. HalfwayStepLRScheduler: Keep initial LR for first half of AL iterations, then drop to fine_tune_lr
            # class HalfwayStepLRScheduler:
            #     def __init__(self, optimizer, current_iteration, total_iterations, initial_lr=1e-4, fine_tune_lr=1e-5):
            #         self.optimizer = optimizer
            #         self.current_iteration = current_iteration
            #         self.total_iterations = total_iterations
            #         self.initial_lr = initial_lr
            #         self.fine_tune_lr = fine_tune_lr
            #         self.halfway_point = total_iterations // 2
            #         # Set initial LR
            #         if current_iteration < self.halfway_point:
            #             target_lr = self.initial_lr
            #         else:
            #             target_lr = self.fine_tune_lr
            #         for param_group in self.optimizer.param_groups:
            #             param_group['lr'] = target_lr
            #     def get_lr(self):
            #         return [group['lr'] for group in self.optimizer.param_groups]
            #     def step(self, *args, **kwargs):
            #         return  # LR is set once at scheduler init, no per-batch update needed
            # scheduler = HalfwayStepLRScheduler(optimizer, current_iteration=iteration, total_iterations=20, initial_lr=1e-4, fine_tune_lr=1e-5)
            # ================================================================

            # If SnP enabled, wrap optimizer.step to add Gaussian noise post-update
            if self.perturb_sigma > 0.0:
                orig_step = optimizer.step
                sigma = float(self.perturb_sigma)
                def snp_step(*step_args, **step_kwargs):
                    loss = orig_step(*step_args, **step_kwargs)
                    try:
                        with torch.no_grad():
                            for p in model.parameters():
                                if p.requires_grad and p.data is not None:
                                    p.add_(torch.randn_like(p) * sigma)
                    except Exception:
                        pass
                    return loss
                optimizer.step = snp_step  # type: ignore

            n_iter = 0
            epoch_losses = []

            for epoch in trange(args.epochs):
                debug(f"Epoch {epoch}")

                result = train(
                    model=model,
                    data_loader=train_data_loader,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    n_iter=n_iter,
                    logger=logger,
                    writer=writer,
                    cbp_trainer=cbp_trainer,
                    epoch=epoch
                )
                
                # Handle both old format (int) and new format (tuple)
                if isinstance(result, tuple):
                    n_iter, epoch_loss = result
                    epoch_losses.append(epoch_loss)
                    try:
                        debug(f"Epoch {epoch} average loss = {epoch_loss:.6f}")
                    except Exception:
                        pass
                else:
                    n_iter = result

                if cbp_trainer and hasattr(cbp_trainer, 'log_epoch_cbp_stats'):
                    cbp_trainer.log_epoch_cbp_stats(epoch)

                    if cbp_trainer.cbp_logger:
                        total_replacements = cbp_trainer.cbp_logger.get_total_replacements()
                        if total_replacements > 0:
                            debug(f"  CBP: Total neuron replacements so far: {total_replacements}")

                        if hasattr(cbp_trainer.cbp_logger, 'replacement_history'):
                            for layer_name, history in cbp_trainer.cbp_logger.replacement_history.items():
                                if history:
                                    layer_replacements = sum(len(h['indices']) for h in history)
                                    if layer_replacements > 0:
                                        debug(f"    {layer_name}: {layer_replacements} replacements")
                    
                if isinstance(scheduler, ExponentialLR):
                    scheduler.step()
            if len(self.models) < args.ensemble_size:
                assert len(self.models) == model_idx
                self.models.append(model)
                if args.cbp and cbp_trainer:
                    if len(self.cbp_trainers) <= model_idx:
                        self.cbp_trainers.extend([None] * (model_idx + 1 - len(self.cbp_trainers)))
                    self.cbp_trainers[model_idx] = cbp_trainer

            self.scalers.append((scaler, features_scaler, None, None))
            
            if epoch_losses:
                self.cbp_stats['epoch_losses'] = epoch_losses
                self.cbp_stats['final_loss'] = epoch_losses[-1] if epoch_losses else None

                if args.cbp:
                    self.cbp_stats['cbp_enabled'] = True
                    self.cbp_stats['iteration'] = iteration
                    debug(f"Training completed for model {model_idx} with CBP enabled")
                    debug(f"  Final loss: {epoch_losses[-1] if epoch_losses else 'N/A'}")
                    debug(f"CBP training completed for iteration {iteration}")

            # Save checkpoint after training each model
            checkpoint_path = os.path.join(save_dir, 'model.pth')
            save_checkpoint(checkpoint_path, model, scaler, features_scaler, None, None, args)
            debug(f"Checkpoint saved to {checkpoint_path}")

            if args.cbp and cbp_trainer:
                cbp_state_path = os.path.join(save_dir, 'cbp_state.pth')
                cbp_state = {
                    'optimizer_state_dict': cbp_trainer.optimizer.state_dict(),
                    'iteration': iteration,
                }
                torch.save(cbp_state, cbp_state_path)
                debug(f"CBP optimizer state saved to {cbp_state_path}")

        if hasattr(self, '_pending_cbp_state') and self._pending_cbp_state:
            debug(f"Cleaning up {len(self._pending_cbp_state)} unused pending CBP state entries")
            self._pending_cbp_state.clear()

    def save_checkpoint(self, iteration: int = 0):
        """Save checkpoints for all ensemble models including CBP state."""
        args = self.chemprop_train_args
        if not hasattr(self, 'models'):
            print("No models to save")
            return

        for model_idx, model in enumerate(self.models):
            save_dir = os.path.join(args.save_dir, f"model_{model_idx}")
            makedirs(save_dir)
            checkpoint_path = os.path.join(save_dir, 'model.pth')
            if model_idx < len(self.scalers):
                scaler, features_scaler, _, _ = self.scalers[model_idx]
            else:
                scaler, features_scaler = None, None
            save_checkpoint(checkpoint_path, model, scaler, features_scaler, None, None, args)
            print(f"✅ Checkpoint saved for model {model_idx} to {checkpoint_path}")

            # Save CBP optimizer state if CBP is enabled
            if args.cbp and hasattr(self, 'cbp_trainer') and self.cbp_trainer:
                cbp_state_path = os.path.join(save_dir, 'cbp_state.pth')
                cbp_state = {
                    'optimizer_state_dict': self.cbp_trainer.optimizer.state_dict(),
                    'iteration': iteration,
                }
                torch.save(cbp_state, cbp_state_path)
                print(f"✅ CBP optimizer state saved to {cbp_state_path}")

    def load_checkpoint(self):
        """Load checkpoints for all ensemble models. Returns (success, last_iteration)."""
        args = self.chemprop_train_args
        models = []
        scalers = []
        loaded_count = 0
        last_iteration = 0

        for model_idx in range(args.ensemble_size):
            save_dir = os.path.join(args.save_dir, f"model_{model_idx}")
            checkpoint_path = os.path.join(save_dir, 'model.pth')

            if os.path.exists(checkpoint_path):
                try:
                    state = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
                    model = load_checkpoint(checkpoint_path, args.device)
                    models.append(model)

                    scaler = state.get('data_scaler', None)
                    features_scaler = state.get('features_scaler', None)
                    scalers.append((scaler, features_scaler, None, None))

                    loaded_count += 1
                    print(f"✅ Loaded checkpoint for model {model_idx} from {checkpoint_path}")

                    cbp_state_path = os.path.join(save_dir, 'cbp_state.pth')
                    if args.cbp and os.path.exists(cbp_state_path):
                        cbp_state = torch.load(cbp_state_path, map_location=args.device, weights_only=False)
                        last_iteration = cbp_state.get('iteration', 0)

                        if not hasattr(self, '_pending_cbp_state'):
                            self._pending_cbp_state = {}
                        self._pending_cbp_state[model_idx] = cbp_state
                        print(f"✅ Loaded CBP state for model {model_idx} (iteration {last_iteration})")

                except Exception as e:
                    print(f"⚠️ Failed to load checkpoint for model {model_idx}: {e}")
                    return False, 0
            else:
                print(f"⚠️ No checkpoint found for model {model_idx} at {checkpoint_path}")
                return False, 0

        if loaded_count == args.ensemble_size:
            self.models = models
            self.scalers = scalers
            print(f"🎉 Successfully loaded all {loaded_count} model checkpoints with scalers")
            return True, last_iteration
        return False, 0

    def close_cbp_trainer(self):
        if self.cbp_trainer and self.cbp_trainer.cbp_logger:
            print("📊 Saving final CBP training summary...")
            self.cbp_trainer.save_cbp_summary()
            self.cbp_trainer = None

    def predict(self, pred_data, batch_size: int = 10000):
        """Return (predictions, uncertainties) arrays for pred_data."""
        args = self.chemprop_predict_args
        train_args = self.chemprop_train_args
        num_tasks = train_args.num_tasks
        task_names = train_args.task_names

        set_features(args, train_args)

        if train_args.features_scaling:
            pred_data.normalize_features(self.scalers[0][0])

        all_preds = []
        all_uncs = []
        total_batches = (len(pred_data) + batch_size - 1) // batch_size
        for i in range(total_batches):
            models = (model for model in self.models)
            scalers = (scaler for scaler in self.scalers)

            start = i * batch_size
            end = min((i + 1) * batch_size, len(pred_data))
            test_data = get_subset_from_idx(pred_data, range(start, end))
            test_data_loader = MoleculeDataLoader(
                dataset=test_data,
                batch_size=train_args.batch_size,
                num_workers=train_args.num_workers
            )
            preds, unc = predict_and_save(
                args=args,
                train_args=train_args,
                test_data=test_data,
                task_names=task_names,
                num_tasks=num_tasks,
                test_data_loader=test_data_loader,
                full_data=pred_data,
                full_to_valid_indices={j: j for j in range(len(pred_data))},
                models=models,
                scalers=scalers,
                num_models=len(self.models),
                return_invalid_smiles=False,
                save_results=False
            )
            all_preds += np.array(preds).ravel().tolist()
            all_uncs += np.array(unc).ravel().tolist()
        return np.array(all_preds), np.array(all_uncs)

    def predict_uncertainty(self, pred_data):
        if self.chemprop_predict_args.uncertainty_method is None and self.chemprop_train_args.dataset_type == "classification":
            preds = np.array(self.predict(pred_data)[0])
            preds = np.array([preds, 1-preds]).T
            return (0.25 - np.var(preds, axis=1)) * 4
        else:
            return self.predict(pred_data)[1]

    def predict_value(self, pred_data):
        return self.predict(pred_data)[0]
    
    def get_cbp_stats(self):
        """Return CBP statistics collected during training"""
        return self.cbp_stats