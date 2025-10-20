#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from molalkit.exe.run import molalkit_run


@pytest.fixture()
def tmpdir(tmp_path):
    d = tmp_path / "snp_test"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_min_cfg(path: Path, extra: dict | None = None):
    cfg = {
        "data_format": "chemprop",
        # keep test fast
        "epochs": 1,
        "batch_size": 8,
        "hidden_size": 32,
        "depth": 1,
        "ffn_num_layers": 1,
        "dropout": 0.0,
        "ensemble_size": 1,
        "loss_function": "binary_cross_entropy",
    }
    if extra:
        cfg.update(extra)
    path.write_text(json.dumps(cfg, indent=2))


def _run(tmpdir: Path, cfg_path: Path, extra_args: list[str]):
    args = [
        "--save_dir", str(tmpdir),
        "--data_public", "bbbp",
        "--task_type", "binary",
        "--metrics", "roc_auc",
        "--split_type", "random",
        "--split_sizes", "0.9", "0.1",
        "--init_size", "2",
        "--model_configs", str(cfg_path),
        "--n_select", "0",
        "--evaluate_stride", "1",
        "--max_iter", "1",
    ] + extra_args
    molalkit_run(args)
    with open(tmpdir / "run_meta.json", "r") as f:
        return json.load(f)


def test_snp_off_with_zero_sigma(tmpdir):
    cfg = tmpdir / "cfg.json"
    _write_min_cfg(cfg)

    meta = _run(tmpdir, cfg, ["--weight_decay", "0.001", "--perturb_sigma", "0.0"])
    # Ensure fields exist and sigma recorded as 0
    assert "model_meta" in meta
    assert float(meta["model_meta"].get("weight_decay", -1)) == pytest.approx(1e-3)
    assert float(meta["model_meta"].get("perturb_sigma", 0.0)) == pytest.approx(0.0)


def test_snp_on_with_sigma(tmpdir):
    cfg = tmpdir / "cfg2.json"
    _write_min_cfg(cfg)

    sigma = 1e-5
    meta = _run(tmpdir, cfg, ["--weight_decay", "1e-4", "--perturb_sigma", str(sigma)])
    assert "model_meta" in meta
    assert float(meta["model_meta"].get("weight_decay", -1)) == pytest.approx(1e-4)
    assert float(meta["model_meta"].get("perturb_sigma", -1)) == pytest.approx(sigma)


