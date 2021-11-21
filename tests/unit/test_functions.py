import pytest
import pycollo
from scipy.interpolate import CubicSpline
import sympy as sym
import cmath as math
from sympy.abc import x, y

def test_cubic_spline():
    """Check that scipy's spline fit matches with the generated Segwise function evaluations"""
    x_data = [0, 1, 2, 3, 4, 5]
    y_data = [0, 1, 0, 1, 2, 0]
    test_points = [0.5, 1.2, 2.4, 3.8, 4.2]
    spline_sym = pycollo.functions.cubic_spline(x, x_data, y_data)
    spline_sci = CubicSpline(x_data, y_data, bc_type="natural")
    for tx in test_points:
        assert math.isclose(spline_sym.subs(x, tx), spline_sci(tx))

def test_cubic_spline_derivative():
    x_data = [0, 1, 2, 3, 4, 5]
    y_data = [0, 1, 0, 1, 2, 0]
    test_points = [0.5, 1.2, 2.4, 3.8, 4.2]
    spline_sym = pycollo.functions.cubic_spline(x, x_data, y_data)
    derivative_sym = sym.diff(spline_sym,x)
    spline_sci = CubicSpline(x_data, y_data, bc_type="natural")
    derivative_sci = spline_sci.derivative()
    for tx in test_points:
        assert math.isclose(derivative_sym.subs(x, tx), derivative_sci(tx))

def test_segwise_asserts_continuity():
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(x,(-x,0.0),(x+1,1.0))
    assert "are not continuous" in str(excinfo.value)

def test_segwise_asserts_derivative_continuity():
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(x,(-x,0.0),(2*x,1.0))
    assert "derivative" in str(excinfo.value)

def test_segwise_no_arguments():
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise()
    assert "zero arguments" in str(excinfo.value)

def test_segwise_one_argument():
    x = sym.Symbol("x")
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(x)
    assert "at least 2 segments" in str(excinfo.value)

def test_segwise_no_symbol():
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(0.1,(-x,0.0),(x+1,1.0))
    assert "symbol" in str(excinfo.value)

def test_segwise_one_segment():
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(x,(-x,0.0))
    assert "at least 2 segments" in str(excinfo.value)

def test_segwise_bad_segment_format():
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(x,(-x,0.0,1.0),(x+1,1.0))
    assert "incorrect format" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(x,(-x),(x+1,1.0))
    assert "incorrect format" in str(excinfo.value)

def test_segwise_nonsequential_bounds():
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(x,(x,1.0),(x,0.0))
    assert "higher upper bound" in str(excinfo.value)

def test_segwise_extra_variables():
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(x,(x,0.0),(x**2+y,1.0))
    assert "other variables" in str(excinfo.value)

def test_segwise_variable_bounds():
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(x,(x,0.0),(x**2,y))
    assert "constant upper bound" in str(excinfo.value)