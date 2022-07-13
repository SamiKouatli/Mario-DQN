#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT

import collections

import gym
import numpy as np


class StackSkipFrame(gym.Wrapper):
    """
    Stack frames to have a sens of motion and skip several frames.
    """

    def __init__(self, env_model, env = None):
        """
        Initialize frame stack and skip layer.

        Parameters
        ----------
        env_model: EnvironmentModel
            Environment model
        env: Env
            Gym environment
        """
        super(StackSkipFrame, self).__init__(env)
        # Number of actions played in a row
        self._skip = env_model.skip_nb_frame
        # Buffer size
        self._buffer_size = env_model.buffer_size
        # Create an empty deque for most recent raw observations (for max pooling across time steps)
        self._buffer = collections.deque(maxlen = self._buffer_size)

    def step(self, action):
        """
        Perform several steps (see skip) with the same action and append last state to buffer.

         Parameters
        ----------
        action: ActType
            Action performed.

        Returns
        -------
        stacked_state: ObsType
            Final state after action performed several times.
        total_reward: float
            Total reward of actions performed several times (see skip).
        done: bool
            Has the episode ended.
        info: dict
            Additional information regarding the reason for episode end.
        """
        total_reward = 0.0
        done = None
        info = None
        obs = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        self._buffer.append(obs)
        stacked_state = np.stack(self._buffer, axis = 2)

        return stacked_state, total_reward, done, info

    def reset(self):
        """
        Reset and fill the buffer with the first observation state several times.
        """
        self._buffer.clear()
        obs = self.env.reset()
        for _ in range(self._buffer_size):
            self._buffer.append(obs)
        stacked_state = np.stack(self._buffer, axis = 2)

        return stacked_state
