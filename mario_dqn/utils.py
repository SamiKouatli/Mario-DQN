#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT

"""
Project utility functions.
"""

from pathlib import Path


def get_project_root():
    """
    Get project path

    Returns
    -------
    root_path: Path
        Project root path
    """
    return Path(__file__).parent.parent


def get_file_from_root(file):
    """
    Get file relative to project root

    Parameters
    ----------
    file: str
        File name

    Returns
    -------
    file_path: Path
        File path
    """
    return get_project_root().joinpath(file)


def get_file_path_from_root(file):
    """
    Get file path relative to project root

    Parameters
    ----------
    file: str
        File name

    Returns
    -------
    file_path: str
        File path in string
    """
    return str(get_file_from_root(file))
