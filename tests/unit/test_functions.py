import pytest
import pycollo
from scipy.interpolate import CubicSpline
import sympy as sym
import cmath as math

def test_cubic_spline():
    """Check that scipy's spline fit matches with the generated Segwise function evaluations"""
    x = sym.Symbol("x")
    x_data = [0, 1, 2, 3, 4, 5]
    y_data = [0, 1, 0, 1, 2, 0]
    test_points = [0.5, 1.2, 2.4, 3.8, 4.2]
    spline_sym = pycollo.functions.cubic_spline(x, x_data, y_data)
    spline_sci = CubicSpline(x_data, y_data, bc_type="natural")
    for tx in test_points:
        assert math.isclose(spline_sym.subs(x, tx), spline_sci(tx))