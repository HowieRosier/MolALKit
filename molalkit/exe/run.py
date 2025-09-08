#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import shutil
import hashlib
from datetime import datetime
import pandas as pd
from molalkit.active_learning.learner import ActiveLearner
from molalkit.exe.args import LearningArgs


def molalkit_run(arguments=None):
    args = LearningArgs().parse_args(arguments)
    logger = args.logger

    # Write lightweight run snapshots for reproducibility
    try:
        os.makedirs(args.save_dir, exist_ok=True)

        # 1) Capture CLI arguments for embedding into run_meta.json
        cli_args = arguments if arguments is not None else sys.argv[1:]

        # 2) Run meta card: key metadata for quick reference
        def _file_meta(path: str):
            meta = None
            if path is not None and os.path.isfile(path):
                try:
                    stat = os.stat(path)
                    # Avoid heavy hashing for very large files (> 500 MB)
                    size_mb = stat.st_size / (1024 * 1024)
                    sha256 = None
                    if size_mb <= 500:
                        h = hashlib.sha256()
                        with open(path, "rb") as rf:
                            for chunk in iter(lambda: rf.read(1024 * 1024), b""):
                                h.update(chunk)
                        sha256 = h.hexdigest()
                    meta = {
                        "exists": True,
                        "size_bytes": stat.st_size,
                        "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "sha256": sha256,
                    }
                except Exception:
                    meta = {"exists": True}
            return meta

        # Force model materialization to read per-model attributes in meta card
        _ = args.models

        selector_model = args.models[0] if len(args.models) > 0 else None
        # Extract minimal model info where available (ChemProp MPNN exposes chemprop_train_args)
        model_meta = {}
        if selector_model is not None:
            if hasattr(selector_model, "chemprop_train_args"):
                ta = selector_model.chemprop_train_args
                model_meta = {
                    "dataset_type": getattr(ta, "dataset_type", None),
                    "loss_function": getattr(ta, "loss_function", None),
                    "cbp_enabled": bool(getattr(ta, "cbp", False)),
                    # L2 / noise
                    "weight_decay": getattr(ta, "weight_decay", None),
                    "perturb_sigma": getattr(selector_model, "perturb_sigma", None),
                    # Model arch
                    "epochs": getattr(ta, "epochs", None),
                    "hidden_size": getattr(ta, "hidden_size", None),
                    "depth": getattr(ta, "depth", None),
                    "ffn_num_layers": getattr(ta, "ffn_num_layers", None),
                    "dropout": getattr(ta, "dropout", None),
                    "batch_size": getattr(ta, "batch_size", None),
                    # CBP params (explicit in meta card)
                    "replacement_rate": getattr(ta, "replacement_rate", None),
                    "decay_rate": getattr(ta, "decay_rate", None),
                    "maturity_threshold": getattr(ta, "maturity_threshold", None),
                    "util_type": getattr(ta, "util_type", None),
                }

        run_meta = {
            "timestamp": datetime.now().isoformat(),
            "save_dir": args.save_dir,
            "seed": args.seed,
            "n_jobs": args.n_jobs,
            "verbose": args.verbose,
            "cli_args": [str(x) for x in cli_args],
            "data_public": getattr(args, "data_public", None),
            "data_path": getattr(args, "data_path", None),
            "data_file_meta": _file_meta(getattr(args, "data_path", None)),
            "split_type": getattr(args, "split_type", None),
            "split_sizes": getattr(args, "split_sizes", None),
            "init_size": getattr(args, "init_size", None),
            "select_method": getattr(args, "select_method", None),
            "s_batch_size": getattr(args, "s_batch_size", None),
            "s_batch_mode": getattr(args, "s_batch_mode", None),
            "s_exploitive_target": getattr(args, "s_exploitive_target", None),
            "forget_method": getattr(args, "forget_method", None),
            "f_batch_size": getattr(args, "f_batch_size", None),
            "evaluate_stride": getattr(args, "evaluate_stride", None),
            "write_traj_stride": getattr(args, "write_traj_stride", None),
            "save_cpt_stride": getattr(args, "save_cpt_stride", None),
            "max_iter": getattr(args, "max_iter", None),
            "model_meta": model_meta,
        }
        with open(os.path.join(args.save_dir, "run_meta.json"), "w") as f_meta:
            json.dump(run_meta, f_meta, indent=2)
    except Exception as e:
        # Do not fail the run due to metadata issues
        logger.debug(f"Run snapshot/meta writing skipped due to: {e}")
    if args.load_checkpoint and os.path.exists("%s/al.pkl" % args.save_dir):
        logger.info("Restart active learning from checkpoint file %s/al.pkl" % args.save_dir)
        active_learner = ActiveLearner.load(path=args.save_dir)
        current_loop = active_learner.current_loop
    else:
        logger.info("Start active learning from scratch")
        active_learner = ActiveLearner(
            save_dir=args.save_dir,
            selector=args.selector,
            forgetter=args.forgetter,
            models=args.models,
            id2datapoints=args.id2datapoints,
            datasets_train=args.datasets_train,
            datasets_pool=args.datasets_pool,
            datasets_val=args.datasets_val,
            metrics=args.metrics,
            top_uidx=args.top_uidx,
            kernel=args.kernels[0],
            detail=args.detail,
        )
        current_loop = 0
        active_learner.evaluate()
    for i in range(current_loop, args.max_iter or 100):
        logger.info("Active learning loop %d" % i)
        for _ in range(args.n_select):
            active_learner.step_select()
            logger.debug("Select step %d" % _)
        if args.f_min_train_size is None or len(active_learner.datasets_train[0]) >= args.f_min_train_size:
            for _ in range(args.n_forget):
                active_learner.step_forget()
                logger.debug("Forget step %d" % _)
        if args.evaluate_stride is not None and i % args.evaluate_stride == 0:
            active_learner.evaluate()
            logger.debug("Evaluate step")
        if i % args.write_traj_stride == 0:
            active_learner.write_traj()
        if args.save_cpt_stride is not None and i % args.save_cpt_stride == 0:
            active_learner.current_loop = i + 1
            active_learner.save(path=args.save_dir, filename="al_temp.pkl", overwrite=True)
            shutil.move(os.path.join(args.save_dir, "al_temp.pkl"), os.path.join(args.save_dir, "al.pkl"))
            logger.info("Save checkpoint file %s/al.pkl" % args.save_dir)
    df = pd.read_csv(f"{args.save_dir}/full.csv")
    df[df["uidx"].isin([data.uidx for data in active_learner.datasets_train[0]])].to_csv(f"{args.save_dir}/train_end.csv", index=False)
    df[df["uidx"].isin([data.uidx for data in active_learner.datasets_pool[0]])].to_csv(f"{args.save_dir}/pool_end.csv", index=False)
