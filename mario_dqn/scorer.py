#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT

from collections import deque

import matplotlib.pyplot as plt


class Scorer:
    """
    Score saver and loader.

    There is two possible mode :
    - Saver which averages the results over severals episodes and stores them in a results file.
    - Loader which loads a results file and plot the results' curve.

    """

    def __init__(self, filepath, buffer_size, scores, steps):
        """
        Initialize the scorer element. This function must be called by Scorer.saver() or Scorer.loader().

        Parameters
        ----------
        filepath: str
            Path to results file.
        buffer_size: int|None
            Score circular buffer size.
        scores: collections.deque|None
            Current scores list.
        steps: collections.deque|None
            Current steps list.
        """
        self._mode = None
        self._scores = scores
        self._steps = steps
        self._buffer_size = buffer_size
        self._filepath = filepath

    @staticmethod
    def loader(filepath):
        """
        Load results file.

        Parameters
        ----------
        filepath: str
            Path to results file.

        Returns
        -------
        loader: Scorer
            Scorer in loader mode.
        """
        loader = Scorer(filepath = filepath,
                        buffer_size = None,
                        scores = None,
                        steps = None)
        loader._mode = _ScorerMode.LOADER
        return loader

    @staticmethod
    def saver(buffer_size, filepath):
        """
        Save results file.

        Parameters
        ----------
        filepath: str
            Path to results file.
        buffer_size: int
            Score circular buffer size.

        Returns
        -------
        loader: Scorer
            Scorer in loader mode.
        """
        saver = Scorer(filepath = filepath,
                       buffer_size = buffer_size,
                       scores = deque(maxlen = buffer_size),
                       steps = deque(maxlen = buffer_size))
        saver._mode = _ScorerMode.SAVER

        # Create results file and write circular buffer size in it
        file = open(filepath, "w")
        file.write(f"{_ScorerConstHelper.BUFFER_SIZE}{buffer_size}{_ScorerConstHelper.RETURN}")
        file.close()
        return saver

    def add(self, episode_num, score, steps):
        """
        Add score and step value of an episodes to circular buffer.

        Parameters
        ----------
        episode_num: int
            Current episode number.
        score: float
            Current episode score.
        steps: float
            Current episode number of steps.
        """
        assert self._mode is _ScorerMode.SAVER

        # Add new scores and steps
        self._scores.append(score)
        self._steps.append(steps)

        # Check episodes has done a loop of circular buffer
        if episode_num % self._buffer_size == 0:
            if episode_num == 0:
                score_mean, steps_mean = score, steps
            else:
                # Compute mean of scores and steps in buffer
                score_mean, steps_mean = self._mean()

            # Append new score and steps in results file
            file = open(self._filepath, "a")
            file.write(f"{_ScorerConstHelper.SCORE}{score_mean}"
                       f"{_ScorerConstHelper.STEPS}{steps_mean}{_ScorerConstHelper.RETURN}")
            file.close()

    def plot(self, image_path):
        """
        Plot results curves for socres and steps. Then save plot.

        Parameters
        ----------
        image_path: str
            Relative path to image from results.
        """
        # Parse file
        score_list, steps_list = self._parse_file()

        # Create x axis
        x = [i * int(self._buffer_size) for i in range(len(score_list))]

        # Create two horizontal plot
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Plot scores and steps
        ax1.plot(x, score_list, color = 'blue')
        ax2.plot(x, steps_list, color = 'orange')

        # Add plots info
        fig.suptitle("Episodes trained")
        ax1.set_xlabel('Episodes')
        ax2.set_xlabel('Episodes')
        ax1.set_ylabel('Scores')
        ax2.set_ylabel('Steps')

        # Save plot file if possible
        if image_path is not None:
            plt.savefig(image_path)

        # Show plot
        plt.show()

    def _mean(self):
        """
        Compute mean of circular scores and steps buffer.

        Returns
        -------
        score_mean: float
            Average score over buffer size.
        steps_mean: float
            Average steps over buffer size.
        """
        assert self._mode is _ScorerMode.SAVER

        # Initialize means
        score_mean = 0
        steps_mean = 0

        # Make addition of every value in buffer
        for i in range(self._buffer_size):
            score_mean += self._scores[i]
            steps_mean += self._steps[i]

        # Mean of each element
        score_mean /= self._buffer_size
        steps_mean /= self._buffer_size

        return score_mean, steps_mean

    def _parse_file(self):
        """
        Parse results file.

        Returns
        -------
        score_list: array
            List of all averaged scores.
        steps_list: array
            List of all averaged steps.
        """
        # Read file
        file = open(self._filepath, "r")
        data = file.read()
        file.close()

        # Initialize array
        score_list = []
        steps_list = []

        # Parse buffer line (first line)
        data = data.split(_ScorerConstHelper.RETURN)
        self._buffer_size = data[0].split(_ScorerConstHelper.SPACE)[-1]

        # Remove first and last line
        data = data[1:-1]

        # Parse the rest of file
        for line in data:
            elements = line.split(_ScorerConstHelper.SEMICOLON)
            score_list.append(float(elements[0].split(_ScorerConstHelper.SPACE)[-1]))
            steps_list.append(float(elements[1].split(_ScorerConstHelper.SPACE)[-1]))

        return score_list, steps_list


class _ScorerMode:
    """
    Scorer mode used.
    """
    SAVER = "saver"
    LOADER = "loader"


class _ScorerConstHelper:
    """
    Scorer parse helper, this class keeps all constants.
    """
    RETURN = "\n"
    SPACE = " "
    SEMICOLON = ";"
    BUFFER_SIZE = "Buffer size: "
    SCORE = "Score: "
    STEPS = ";Steps: "
