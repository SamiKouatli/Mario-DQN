#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT


import abc

import yaml


class Parser(abc.ABC):
    """Abstract class for YAML parser."""

    def __init__(self, path):
        """Open given file, stop project if error while parsing.

        Parameters
        ----------
        path: Path
            File path
        """
        file = open(path, "r")
        try:
            self.raw = yaml.safe_load(file)
        except BaseException as exc:
            print(f"[Parser] Error when loading {path}: {exc}")
            exit()
        file.close()

    @staticmethod
    def _get_section(dictionary, section):
        """
        Get dictionary section, stop project if error while parsing.

        Parameters
        ----------
        dictionary: dict
            Dictionary from YAML parsing
        section :str
            Section in dictionary

        Returns
        -------
        value: int|float|str|array
            Value in dictionnary
        """
        value = dictionary.get(section, None)
        if value is None:
            print(f"[Parser] Missing {section} in {dictionary}")
            exit()

        return value

    @abc.abstractmethod
    def _print_config(self):
        """
        Abstract print config hyperparameters.
        """
        pass
