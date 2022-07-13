#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from config.paths import VIDEO_SAVE_PATH
from mario_dqn.environment.environment_parser import EnvironmentParser
from mario_dqn.environment.wrapers.preprocess_frame import PreprocessFrame
from mario_dqn.environment.wrapers.stack_skip_frame import StackSkipFrame


class Environment:
    """
    Processed Super Mario Bros (NES) environment with Gym.
    """

    def __init__(self, is_training, is_recording, save_path, is_verbose):
        """
        Initialize environment: get hyperparameters, preprocess and stack frames.

        Parameters
        ----------
        is_training: bool
            Is the model in training or inference.
        is_recording: bool
            Is the recording activated.
        save_path: str
            Path to save folder.
        is_verbose: bool
            Is verbose enabled.
        """
        # Get environment model from parser
        self.model = EnvironmentParser(is_verbose = is_verbose).content

        # Create a new environment
        env = gym_super_mario_bros.make(self.model.env)

        # Record video
        if is_recording:
            # FIXME(SKO): Record warnings come from gym and
            #  are linked to: https://github.com/openai/gym/issues/2905
            env = gym.wrappers.RecordVideo(env,
                                           f"{save_path}/{VIDEO_SAVE_PATH}",
                                           episode_trigger = lambda x: self._episode_trigger(
                                                   recorded_episode_each = self.model.recorded_episode_each,
                                                   is_training = is_training,
                                                   step = x))

        # Preprocess frame (resize, grayscale and normalize pixel) to lower computational complexity
        env = PreprocessFrame(env_model = self.model, env = env)

        # Stack frames to have a sens of motion and skip several frames
        env = StackSkipFrame(env_model = self.model, env = env)

        # Initialize the environment
        self.env = JoypadSpace(env, RIGHT_ONLY)

    @staticmethod
    def _episode_trigger(recorded_episode_each, is_training, step):
        """
        Callback on each episode to know if recording is required or not.

        Parameters
        ----------
        recorded_episode_each: int
            Episode record period (when in training).
        is_training: bool
            Is the model in training or inference.
        step: int
            Current episode.

        Returns
        -------
        record: bool
            Record episode or not.
        """
        if is_training:
            return step % recorded_episode_each == 0
        else:
            return True
