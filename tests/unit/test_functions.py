import pytest
import pycollo
from scipy.interpolate import CubicSpline
import sympy as sym
import cmath as math
from sympy.abc import x, y
import numpy as np
s = pycollo.functions.Segwise.s

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

def test_cyclic_spline():
    x_data = [0, 2, 3, 4, 5]
    y_data = [0, 0, 1, 2, 0]
    test_points = [1.2, 2.4, 3.8, 4.2, 6.8, 10.3]
    spline_sym = pycollo.functions.cubic_spline(x, x_data, y_data, "periodic")
    spline_sci = CubicSpline(x_data, y_data, bc_type="periodic")
    for tx in test_points:
        assert math.isclose(spline_sym.subs(x, tx), spline_sci(tx%5))

def test_segwise_continuity_check():
    assert not pycollo.functions.Segwise(x,(-s,0.0),(s+1,1.0)).check_continuity()

def test_segwise_asserts_derivative_continuity():
    assert not pycollo.functions.Segwise(x,(-s,0.0),(2*s,1.0)).check_continuity()

def test_segwise_no_arguments():
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise()
    assert "zero arguments" in str(excinfo.value)

def test_segwise_one_argument():
    x = sym.Symbol("x")
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(x)
    assert "at least 2 segments" in str(excinfo.value)

def test_segwise_no_expression():
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(0.1,(-s,0.0),(s+1,1.0))
    assert "sympy expression" in str(excinfo.value)

def test_segwise_one_segment():
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(x,(-s,0.0))
    assert "at least 2 segments" in str(excinfo.value)

def test_segwise_bad_segment_format():
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(x,(-s,0.0,1.0),(s+1,1.0))
    assert "incorrect format" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(x,(-s),(s+1,1.0))
    assert "incorrect format" in str(excinfo.value)

def test_segwise_nonsequential_bounds():
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(x,(s,1.0),(s,0.0))
    assert "higher upper bound" in str(excinfo.value)

def test_segwise_extra_variables():
    assert not pycollo.functions.Segwise(x,(s,0.0),(s**2+y,1.0)).check_continuity()

def test_segwise_variable_bounds():
    with pytest.raises(ValueError) as excinfo:
        pycollo.functions.Segwise(x,(s,0.0),(s**2,y))
    assert "constant upper bound" in str(excinfo.value)

def test_segwise_equispaced_detection():
    nonequispaced = pycollo.functions.Segwise(x,(s,0.0),(s,1.0),(s,3.0))
    assert not nonequispaced.equispaced
    equispaced = pycollo.functions.Segwise(x,(s,0.0),(s,1.0),(s,2.0))
    assert equispaced.equispaced

def test_segwise_infinite_bounds():
    # don't want functions with infinite bounds to be considered equispaced
    seg = pycollo.functions.Segwise(x,(s,0.0),(s,sym.oo))
    assert not seg.equispaced
    # some people might use np.inf instead
    npseg = pycollo.functions.Segwise(x, (s, 0.0), (s, np.inf))
    assert not npseg.equispaced

def test_segwise_with_expression():
    seg = pycollo.functions.Segwise(x+y,(s**3,0.0),(s**2,np.inf))
    assert seg.subs(x, -1.0).subs(y, -1.0) == -8.0
    assert seg.subs(y,1.0).subs(x,1.0) == 4.0

def test_cyclic_segwise_wrap_check():
    seg = pycollo.functions.CyclicSegwise(x,(s,1.0),(0.5*s**2+0.5,2.0))
    assert not seg.check_continuity()
    seg = pycollo.functions.CyclicSegwise(x,(sym.sin(s),1.0),(sym.sin(s),math.tau))
    assert seg.check_continuity()