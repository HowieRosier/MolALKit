#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob


MODEL_DIR = os.path.dirname(__file__)
file_list = glob.glob(os.path.join(MODEL_DIR, "*Config"))
AVAILABLE_MODELS = sorted([os.path.basename(file) for file in file_list])
