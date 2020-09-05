"""Tests for sanitising and processing user-supplied guesses.
"""
import numpy as np
import pytest
import sympy as sym
import sympy.physics.mechanics as me

import pycollo


def test_create_guess():
    """Test whether a :obj:`PhaseGuess` can be instantiated sucessfully."""
    _ = pycollo.PhaseGuess()
    _ = pycollo.PhaseGuess()


def test_1():
    """Dummy test 1."""
    assert(1 == 1)


def test_2():
    """Dummy test 2."""
    assert(1 != 2)
