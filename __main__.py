#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT

import argparse

from config.paths import FROM_RESULT_TO_IMAGE_PATH
from mario_dqn.runner.runner import Runner
from mario_dqn.scorer import Scorer


class _Action:
    """
    Class used to define Action constants.
    """
    TRAIN = "train"
    INFER = "infer"
    PLOT = "plot"


def _parse_args():
    """
    Parse arguments given in command.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("action",
                        type = str,
                        choices = [_Action.TRAIN, _Action.INFER, _Action.PLOT],
                        help = "Train model or infer a known model. Can also plot the results of a trained model.")
    parser.add_argument("--path",
                        type = str,
                        help = "Path to pretrained model if action is train or infer. Path to results.txt if action "
                               "is plot. Value has to be filled when action is infer.")
    parser.add_argument("--record",
                        type = bool,
                        default = True,
                        help = "Is video recording. In training: episodes recorded are in ./config/runner.yaml. "
                               "In inference: record the only episode")
    parser.add_argument("--verbose",
                        type = bool,
                        default = False,
                        help = "Is detailed execution info required.")

    args = parser.parse_args()

    return args.action, args.path, args.record, args.verbose


def main():
    """
    Get arguments given in command-line and act accordingly.
    """
    # Get arguments given
    action, path, is_recording, is_verbose = _parse_args()
    if action is _Action.PLOT:
        is_recording = False

    # Print arguments
    if is_verbose:
        print("***************** COMMAND LINE CONFIGURATION *****************")
        print(f"Action is: {action}")
        print(f"Path: {path}")
        print(f"Recording: {is_recording}")

    # Run chosen action
    if action == _Action.TRAIN:
        Runner().train(pretrained_model_path = path,
                       is_recording = is_recording,
                       is_verbose = is_verbose)
    elif action == _Action.INFER:
        Runner().infer(pretrained_model_path = path,
                       is_recording = is_recording,
                       is_verbose = is_verbose)
    elif action == _Action.PLOT:
        Scorer.loader(filepath = path).plot(f"{path}{FROM_RESULT_TO_IMAGE_PATH}")


if __name__ == '__main__':
    main()
