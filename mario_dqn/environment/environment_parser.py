#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT

from config.paths import ENVIRONMENT_CONFIG_PATH
from mario_dqn.environment.environment_model import EnvironmentModel
from mario_dqn.parser import Parser
from mario_dqn.utils import get_file_path_from_root


class EnvironmentParser(Parser):
    """
    Parse environment hyperparameters (in config) an keep them inside a model.
    """

    def __init__(self, is_verbose):
        """
        Parse environment configuration, save and print configuration.

        Parameters
        ----------
        is_verbose: bool
            Is verbose enabled.
        """
        # Get environment config file
        path = get_file_path_from_root(ENVIRONMENT_CONFIG_PATH)
        super(EnvironmentParser, self).__init__(path)

        # Get section with information on the game environment
        game_section = self._get_section(self.raw, _EnvironmentParserHelper.GAME_SECTION_NAME)
        # Get game environment name used by gym
        env = self._get_section(game_section, _EnvironmentParserHelper.ENV_SECTION_NAME)

        # Get input frame dimension of the game environment
        input_frame_section = self._get_section(game_section, _EnvironmentParserHelper.FRAME_SECTION_NAME)
        input_width = self._get_section(input_frame_section, _EnvironmentParserHelper.WIDTH_SECTION_NAME)
        input_height = self._get_section(input_frame_section, _EnvironmentParserHelper.HEIGHT_SECTION_NAME)
        input_color_channel = self._get_section(input_frame_section,
                                                _EnvironmentParserHelper.COLOR_CHANNEL_SECTION_NAME)

        # Pixel cropped from frame in each direction
        crop_section = self._get_section(self.raw, _EnvironmentParserHelper.CROP_SECTION_NAME)
        crop_left = self._get_section(crop_section, _EnvironmentParserHelper.LEFT_SECTION_NAME)
        crop_right = self._get_section(crop_section, _EnvironmentParserHelper.RIGHT_SECTION_NAME)
        crop_up = self._get_section(crop_section, _EnvironmentParserHelper.UP_SECTION_NAME)
        crop_down = self._get_section(crop_section, _EnvironmentParserHelper.DOWN_SECTION_NAME)

        # Get skip each nb frame
        skip_nb_frame = self._get_section(self.raw, _EnvironmentParserHelper.SKIP_NB_SECTION_NAME)

        # Get buffer size
        buffer_size = self._get_section(self.raw, _EnvironmentParserHelper.BUFFER_SIZE_SECTION_NAME)

        # Get downscale target size
        downscale_size = self._get_section(self.raw, _EnvironmentParserHelper.DOWNSCALE_SECTION_NAME)

        # Get pixel max value
        pixel_max = self._get_section(self.raw, _EnvironmentParserHelper.PIXEL_MAX_SECTION_NAME)

        # Get episode record period
        recorded_episode_each = self._get_section(self.raw, _EnvironmentParserHelper.RECORDED_EPISODES_SECTION_NAME)

        # Set parser content
        self.content = EnvironmentModel(env = env,
                                        input_width = input_width,
                                        input_height = input_height,
                                        input_color_channel = input_color_channel,
                                        downscale_size = downscale_size,
                                        crop_left = crop_left,
                                        crop_right = crop_right,
                                        crop_up = crop_up,
                                        crop_down = crop_down,
                                        skip_nb_frame = skip_nb_frame,
                                        buffer_size = buffer_size,
                                        pixel_max = pixel_max,
                                        recorded_episode_each = recorded_episode_each)

        # Print config
        if is_verbose:
            self._print_config()

    def _print_config(self):
        """
        Print environment hyperparameters.
        """
        content = self.content
        print("***************** ENVIRONMENT CONFIGURATION *****************")
        print(f"Environment: {content.env}")

        w, h, c = content.get_frame_dimension()
        print(f"Frame dimension:\nWidth: {w}\nHeight: {h}\nChannel (color): {c}")
        print(f"Pixel max value: {content.pixel_max}")

        print(f"Downscale to size: {content.downscale_size}x{content.downscale_size}x1")

        u, d, l, r = content.get_crop_dimension()
        print(f"Crop dimension:\nUp: {u}\nDown: {d}\nLeft: {l}\nRight: {r}")

        print(f"Skip each nb frame: {content.skip_nb_frame}")
        print(f"Buffer size: {content.buffer_size}")
        print(f"Record each {content.recorded_episode_each} episodes")


class _EnvironmentParserHelper:
    """
    Environment parse helper, this class keeps all constants.
    """
    GAME_SECTION_NAME = "game"
    ENV_SECTION_NAME = "env"
    FRAME_SECTION_NAME = "frame"
    WIDTH_SECTION_NAME = "width"
    HEIGHT_SECTION_NAME = "height"
    COLOR_CHANNEL_SECTION_NAME = "color_channel"
    DOWNSCALE_SECTION_NAME = "downscale_size"
    CROP_SECTION_NAME = "crop"
    LEFT_SECTION_NAME = "left"
    RIGHT_SECTION_NAME = "right"
    UP_SECTION_NAME = "up"
    DOWN_SECTION_NAME = "down"
    SKIP_NB_SECTION_NAME = "skip_nb"
    BUFFER_SIZE_SECTION_NAME = "buffer_size"
    PIXEL_MAX_SECTION_NAME = "pixel_max"
    RECORDED_EPISODES_SECTION_NAME = "episodes_recorded_each"
