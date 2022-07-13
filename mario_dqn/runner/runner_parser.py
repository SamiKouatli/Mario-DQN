#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT

from config.paths import RUNNER_CONFIG_PATH
from mario_dqn.parser import Parser
from mario_dqn.runner.runner_model import RunnerModel
from mario_dqn.utils import get_file_path_from_root


class RunnerParser(Parser):
    """
    Parse runner hyperparameters (in config) an keep them inside a model.
    """

    def __init__(self, is_training, pretrained, is_verbose):
        """
        Parse environment configuration, save and print configuration.

        Parameters
        ----------
        is_training: bool
            Is the model in training or inference.
        pretrained: bool
            Is model pretrained.
        is_verbose: bool
            Is verbose enabled.
        """
        path = get_file_path_from_root(RUNNER_CONFIG_PATH)
        super(RunnerParser, self).__init__(path)

        # Get parameters
        self._is_training = is_training
        self._pretrained = pretrained

        # Parse config file
        memory_size = self._get_section(self.raw, _RunnerParserHelper.MEMORY_SIZE_SECTION_NAME)
        batch_size = self._get_section(self.raw, _RunnerParserHelper.BATCH_SIZE_SECTION_NAME)
        num_episodes = self._get_section(self.raw, _RunnerParserHelper.NUM_EPISODES_SECTION_NAME)
        gamma = self._get_section(self.raw, _RunnerParserHelper.GAMMA_SECTION_NAME)
        learning_rate = self._get_section(self.raw, _RunnerParserHelper.LEARNING_RATE_SECTION_NAME)
        explo_max = self._get_section(self.raw, _RunnerParserHelper.EXPLORATION_MAX_SECTION_NAME)
        explo_min = self._get_section(self.raw, _RunnerParserHelper.EXPLORATION_MIN_SECTION_NAME)
        explo_decay = self._get_section(self.raw, _RunnerParserHelper.EXPLORATION_DECAY_SECTION_NAME)
        score_buffer_size = self._get_section(self.raw, _RunnerParserHelper.SCORE_BUFFER_SIZE_SECTION_NAME)

        # Set parser content
        self.content = RunnerModel(max_memory_size = memory_size,
                                   batch_size = batch_size,
                                   num_episodes = num_episodes,
                                   gamma = gamma,
                                   learning_rate = learning_rate,
                                   exploration_max = explo_max,
                                   exploration_min = explo_min,
                                   exploration_decay = explo_decay,
                                   score_buffer_size = score_buffer_size)

        # Print config
        if is_verbose:
            self._print_config()

    def _print_config(self):
        """
        Print runner hyperparameters.
        """
        content = self.content
        print("***************** RUNNER CONFIGURATION *****************")
        if self._is_training:
            print(f"Number of episodes: {content.num_episodes}")
            print(f"Batch size: {content.batch_size}")
            print(f"Learning rate: {content.learning_rate}")
            print(f"Gamma: {content.gamma}")
            print(f"Memory size for experience replay: {content.max_memory_size}")
            print(f"Score buffer size: {content.score_buffer_size}")

            e_max, e_min, e_decay = content.get_exploration_rate()
            print(f"Epsilon:\nMax: {e_max}\nMin: {e_min}\nDecay: {e_decay}")
        else:
            print(f"Project is in infer mode")


class _RunnerParserHelper:
    """
    Runner parse helper, this class keeps all constants.
    """
    MEMORY_SIZE_SECTION_NAME = "max_memory_size"
    BATCH_SIZE_SECTION_NAME = "batch_size"
    NUM_EPISODES_SECTION_NAME = "num_episodes"
    GAMMA_SECTION_NAME = "gamma"
    LEARNING_RATE_SECTION_NAME = "learning_rate"
    EXPLORATION_MAX_SECTION_NAME = "exploration_max"
    EXPLORATION_MIN_SECTION_NAME = "exploration_min"
    EXPLORATION_DECAY_SECTION_NAME = "exploration_decay"
    SCORE_BUFFER_SIZE_SECTION_NAME = "score_buffer_size"
