import sympy as sym
from scipy.interpolate import CubicSpline

class Segwise(sym.Function):
    """Piecewise function for sequential linear segments.

    >>> Segwise(bounding_symbol, (equation,upper_bound),(equation_2, upper_bound_2)...)"""
    nargs=None
    @classmethod
    def eval(cls, x, *args):
        if x.is_Number:
            for eq,ub in args:
                if ub>=x:
                    return eq

def cubic_spline(x,x_data,y_data):
    """Create a cubic spline"""
    spline = CubicSpline(x_data, y_data, bc_type="natural")
    return Segwise(x,*[(sum(spline.c[m, i] * (x - px)**(3-m) for m in range(4)),spline.x[i+1]) for i,px in enumerate(spline.x[:-1])])