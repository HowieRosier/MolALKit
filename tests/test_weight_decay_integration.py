#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

import pytest

from molalkit.exe.run import molalkit_run


@pytest.fixture()
def tmpdir(tmp_path):
    d = tmp_path / "wd_test"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_min_chemprop_config(path: Path, extra: dict | None = None):
    cfg = {
        "data_format": "chemprop",
        # keep training zero-cost for tests
        "epochs": 0,
        "batch_size": 16,
        "hidden_size": 64,
        "depth": 1,
        "ffn_num_layers": 1,
        "dropout": 0.0,
        "ensemble_size": 1,
        "loss_function": "binary_cross_entropy",
    }
    if extra:
        cfg.update(extra)
    path.write_text(json.dumps(cfg, indent=2))


def _run_with_args(tmpdir: Path, model_cfg: Path, extra_args: list[str]):
    args = [
        "--save_dir", str(tmpdir),
        "--data_public", "bbbp",
        "--task_type", "binary",
        "--metrics", "roc_auc",
        "--split_type", "random",
        # ensure non-empty validation split to avoid metric errors
        "--split_sizes", "0.9", "0.1",
        "--init_size", "2",
        "--model_configs", str(model_cfg),
        # no select/forget to minimize runtime
        "--n_select", "0",
        "--evaluate_stride", "1",
        # single AL iteration (initial evaluate still runs)
        "--max_iter", "1",
    ] + extra_args
    molalkit_run(args)
    with open(tmpdir / "run_meta.json", "r") as f:
        return json.load(f)


def test_cli_weight_decay_passes_to_trainargs_and_run_meta(tmpdir):
    cfg = tmpdir / "model_cfg_cli.json"
    _write_min_chemprop_config(cfg)

    wd = "0.001"
    meta = _run_with_args(tmpdir, cfg, ["--weight_decay", wd])

    assert "model_meta" in meta
    # run_meta writes numeric types; compare as float
    assert pytest.approx(float(wd)) == float(meta["model_meta"].get("weight_decay", -1))


def test_config_weight_decay_overrides_cli(tmpdir):
    cfg = tmpdir / "model_cfg_cfgwins.json"
    cfg_wd = 1e-5
    _write_min_chemprop_config(cfg, extra={"weight_decay": cfg_wd})

    cli_wd = "0.001"
    meta = _run_with_args(tmpdir, cfg, ["--weight_decay", cli_wd])

    assert "model_meta" in meta
    assert pytest.approx(cfg_wd) == float(meta["model_meta"].get("weight_decay", -1))


