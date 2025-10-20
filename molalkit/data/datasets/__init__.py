#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob


DATA_DIR = os.path.dirname(__file__)
file_list = glob.glob(os.path.join(DATA_DIR, "*.csv"))
AVAILABLE_DATASETS = sorted([os.path.basename(file) for file in file_list])
