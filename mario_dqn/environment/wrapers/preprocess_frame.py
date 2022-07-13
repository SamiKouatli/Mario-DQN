#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT

import cv2
import gym
import numpy as np
from skimage import transform
from skimage.color import rgb2gray


class PreprocessFrame(gym.ObservationWrapper):
    """
    Resize, grayscale and normalize frame to lower computational complexity.
    """

    def __init__(self, env_model, env = None):
        """
        Initialize frame pre-processor

        Parameters
        ----------
        env_model: EnvironmentModel
            Environment model.
        env: Env
            Gym environment.
        """
        super(PreprocessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low = 0.0,
                                                high = 1.0,
                                                shape = (env_model.downscale_size, env_model.downscale_size, 1),
                                                dtype = np.float32)

        self._model = env_model

    def observation(self, obs):
        """
        Returns a modified observation.

        Parameters
        ----------
        obs: ObsType
            Observation space.

        Returns
        -------
        resized: ndarray
            Preprocessed version of the input.
        """
        return self.process(obs)

    def process(self, frame):
        """
        Preprocess frame: resize, grayscale and normalize it.

        Parameters
        ----------
        frame: ObsType
            Observation space.

        Returns
        -------
        resized: ndarray
            Preprocessed version of the input.
        """
        # Get frame dimension
        w, h, c = self._model.get_frame_dimension()

        # Program works only with the predefined size,
        # hence we compute number of bytes in a frame
        bytes_nb = w * h * c
        if frame.size != bytes_nb:
            print(f"[ProcessFrame] Wrong frame size {bytes_nb}")
            exit()

        # Grayscale frame
        grayed = rgb2gray(frame)

        # Reshape image
        reshaped = np.reshape(grayed, [h, w, 1]).astype(np.float32)

        # Get scale and crop constants
        downscale_size = self._model.downscale_size
        up, down, left, right = self._model.get_crop_dimension()

        # Downscale
        resized_width = downscale_size + left + right
        resized_height = downscale_size + up + down
        resized_screen = cv2.resize(reshaped, (resized_width, resized_height), interpolation = cv2.INTER_AREA)

        # Crop
        cropped = resized_screen[up:(resized_height - down), left:(resized_width - right)]

        # Normalize Pixel Values
        normalized = cropped / float(self._model.pixel_max)

        # Reshape to final size
        out = transform.resize(normalized, [downscale_size, downscale_size])

        return out
