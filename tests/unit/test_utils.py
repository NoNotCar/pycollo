"""Tests for the utils module."""

import casadi as ca
import pytest
import sympy as sym
import math

from pycollo.utils import sympy_to_casadi
from pycollo import functions


@pytest.mark.parametrize("in_mapping", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_sympy_to_casadi(in_mapping, utils):
    """Test utility function for converting Sympy expressions to CasADi ones.

    >>> x, y = sym.symbols("x, y")
    >>> X = ca.SX.sym("x")
    >>> Y = ca.SX.sym("y")
    >>> xy = sym.Matrix([x, y])

    A matrix consisting of some arbitrary expressions is created to showcase
    the differences in syntax between Sympy's and CasADi's internal
    mathematical functions.

    >>> e = sym.Matrix([x * sym.sqrt(y), sym.sin(x + y), abs(x - y)])
    >>> XY = ca.vertcat(X, Y)
    >>> E = sympy_to_casadi(e, xy, XY)

    Display the Sympy expression.

    >>> print(e)
    Matrix([[x*sqrt(y)], [sin(x + y)], [Abs(x - y)]])

    Display the CasADi expression and note that it is mathematically equivalent
    to the Sympy one.

    >>> print(E)
    [(x*sqrt(y)), sin((x+y)), fabs((x-y))]

    """
    x_sympy = sym.Symbol("x")
    y_sympy = sym.Symbol("y")
    x_casadi = ca.SX.sym("x")
    y_casadi = ca.SX.sym("y")
    sympy_vars = [x_sympy, y_sympy]
    casadi_vars = [x_casadi, y_casadi]
    sym_mapping = {var_sym: var_ca 
                       for var_sym, var_ca, use in zip(sympy_vars, casadi_vars, in_mapping)
                       if use}

    expr_sym = sym.Matrix([x_sympy * sym.sqrt(y_sympy),
                           sym.sin(x_sympy + y_sympy),
                           abs(x_sympy - y_sympy)])
    expr_ca, sym_mapping = sympy_to_casadi(expr_sym, sym_mapping)

    assert str(expr_sym) == "Matrix([[x*sqrt(y)], [sin(x + y)], [Abs(x - y)]])"
    assert str(expr_ca) == "[(x*sqrt(y)), sin((x+y)), fabs((x-y))]"
    utils.assert_ca_expr_identical(expr_ca, expr_sym)


def test_cubic_spline_translation():
    x_sympy = sym.Symbol("x")
    x_casadi = ca.SX.sym("x")
    spline_sym = functions.cubic_spline(x_sympy,[0,1,2,3,4,5],[0,1,0,1,2,0])
    sym_mapping = {x_sympy: x_casadi}
    spline_ca, sym_mapping = sympy_to_casadi(spline_sym, sym_mapping)
    spline_fun = ca.Function('fun', [x_casadi], [spline_ca])
    test_points = [0.5,1.2,2.4,3.8,4.2]
    for tx in test_points:
        assert math.isclose(spline_sym.subs(x_sympy,tx),float(spline_fun(tx)))


def test_cyclic_spline_translation():
    x_sympy = sym.Symbol("x")
    x_casadi = ca.SX.sym("x")
    spline_sym = functions.cubic_spline(x_sympy,[1,2,3,4,5],[1,0,1,2,1],"periodic")
    sym_mapping = {x_sympy: x_casadi}
    spline_ca, sym_mapping = sympy_to_casadi(spline_sym, sym_mapping)
    spline_fun = ca.Function('fun', [x_casadi], [spline_ca])
    test_points = [1.2,2.4,3.8,4.2]
    test_points.extend([tp+4 for tp in test_points])
    for tx in test_points:
        assert math.isclose(spline_sym.subs(x_sympy,tx),float(spline_fun(tx)))
