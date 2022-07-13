#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT


class EnvironmentModel:
    """
    Environment model built from environment hyperparameters (in config).
    """

    def __init__(self,
                 env,
                 input_width,
                 input_height,
                 input_color_channel,
                 downscale_size,
                 crop_left,
                 crop_right,
                 crop_up,
                 crop_down,
                 skip_nb_frame,
                 buffer_size,
                 pixel_max,
                 recorded_episode_each):
        """
        Save environment hyperparemeters in model attributes.

        Parameters
        ----------
        env: str
            Game environment to play.
        input_width: int
            Game frame width.
        input_height: int
            Game frame height.
        input_color_channel: int
            Game frame number of color channel.
        downscale_size: int
            Output size of frame's width and height after processing.
        crop_left: int
            Number of pixels to crop on the left size of frame.
        crop_right: int
            Number of pixels to crop on the right size of frame.
        crop_up: int
            Number of pixels to crop on the upper size of frame.
        crop_down: int
            Number of pixels to crop on the lower size of frame.
        skip_nb_frame: int
            Number of frame to skip at each action.
        buffer_size: int
            Number of frame to stack for experience replay.
        pixel_max: int
            Max pixel value.
        recorded_episode_each: int
            Period at which we record episodes.
        """
        self.env = env
        self._input_width = input_width
        self._input_height = input_height
        self._input_color_channel = input_color_channel
        self.downscale_size = downscale_size
        self._crop_left = crop_left
        self._crop_right = crop_right
        self._crop_up = crop_up
        self._crop_down = crop_down
        self.skip_nb_frame = skip_nb_frame
        self.buffer_size = buffer_size
        self.pixel_max = pixel_max
        self.recorded_episode_each = recorded_episode_each

    def get_frame_dimension(self):
        """
        Get frame dimensions.

        Returns
        -------
        input_width: int
            Input width.
        input_height: int
            Input height.
        input_color_channel: int
            Input color channel.
        """
        return self._input_width, self._input_height, self._input_color_channel

    def get_crop_dimension(self):
        """
        Get crop frame dimensions

        Returns
        -------
        crop_up: int
            Crop from top of image.
        crop_down: int
            Crop from bottom of image.
        crop_left: int
            Crop from left of image.
        crop_right: int
            Crop from right of image.
        """
        return self._crop_up, self._crop_down, self._crop_left, self._crop_right
