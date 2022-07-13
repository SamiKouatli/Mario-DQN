#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT

from collections import deque

import numpy as np


class Memory:
    """
    Memory of agent, this is used for experience replay.
    """

    def __init__(self, max_size):
        """
        Initialize memory of agent.

        Parameters
        ----------
        max_size: int
            Agent maximum memory.
        """
        self.buffer = deque(maxlen = max_size)

    def add(self, state, action, reward, next_state, done):
        """
        Add experience in memory.

        Parameters
        ----------
        state: State
            Environment state.
        action: Action
            Agent action chosen.
        reward: float
            Agent reward for action.
        next_state: State
            Environment next state.
        done: bool
            Has the Agent reached the end of Environment.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Get sample from memory

        Parameters
        ----------
        batch_size: int
            Size of batch to sample.

        Returns
        ----------
        state: State array
            Sample of Environment states.
        action: Action array
            Sample of Agent action chosen.
        reward: float array
            Sample of Agent reward for action.
        next_state: State array
            Sample of Environment next state.
        done: bool array
            Sample of "has the Agent reached the end of Environment".
        """
        index = np.random.choice(np.arange(len(self.buffer)),
                                 size = batch_size,
                                 replace = False)

        sample = [self.buffer[i] for i in index]
        states = np.array([each[0] for each in sample], ndmin = 3, dtype = object).astype(np.float32)
        actions = np.array([each[1] for each in sample])
        rewards = np.array([each[2] for each in sample])
        next_states = np.array([each[3] for each in sample], ndmin = 3, dtype = object).astype(np.float32)
        dones = np.array([each[4] for each in sample])

        return states, actions, rewards, next_states, dones
