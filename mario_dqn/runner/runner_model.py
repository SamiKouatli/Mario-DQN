#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT

class RunnerModel:
    """
    Runner model built from environment hyperparameters (in config).
    """

    def __init__(self,
                 batch_size,
                 num_episodes,
                 max_memory_size,
                 learning_rate,
                 exploration_max,
                 exploration_min,
                 exploration_decay,
                 gamma,
                 score_buffer_size):
        """
        Save environment hyperparemeters in model attributes.

        Parameters
        ----------
        batch_size: int
            Batch size
        num_episodes: int
            Number of episodes to play.
        max_memory_size: int
            Agent maximum memory size for experience replay.
        learning_rate: float
            Learning rate (alpha) of agent.
        exploration_max: float
            Exploration (epsilon) max of agent.
        exploration_min: float
            Exploration (epsilon) min of agent.
        exploration_decay: float
            Exploration (epsilon) decay rate of agent.
        gamma: float
            Discout factor (gamma) of agent.
        score_buffer_size: int
            Score circular buffer size.

        """
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.max_memory_size = max_memory_size
        self.learning_rate = learning_rate
        self._exploration_max = exploration_max
        self._exploration_min = exploration_min
        self._exploration_decay = exploration_decay
        self.gamma = gamma
        self.score_buffer_size = score_buffer_size

    def get_exploration_rate(self):
        """
        Get exploration rate data.

        Returns
        -------
        exploration_max: float
            Max exploration rate.
        exploration_min: float
            Min exploration rate.
        exploration_decay: float
            Decay exploration rate.
        """
        return self._exploration_max, self._exploration_min, self._exploration_decay
