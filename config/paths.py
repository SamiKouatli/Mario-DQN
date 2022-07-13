#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT


"""
Project constants.
"""

# Hyperparameters config folder
_CONFIG_PATH = "config"

# Environment hyperparameters config file
ENVIRONMENT_CONFIG_PATH = f"{_CONFIG_PATH}/environment.yaml"

# Runner hyperparameters config file
RUNNER_CONFIG_PATH = f"{_CONFIG_PATH}/runner.yaml"

# Save folder
SAVE_PATH = "save"

# Model save relative path
NN_MODEL_SAVE_PATH = "weights.h5"

# Video save relative path
VIDEO_SAVE_PATH = "video/"

# Results save relative path
RESULTS_SAVE_PATH = "results.txt"

# Results plot save relative path
FROM_RESULT_TO_IMAGE_PATH = "/../results_plot.png"
