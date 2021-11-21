import sympy as sym
from scipy.interpolate import CubicSpline
import cmath as math

class Segwise(sym.Function):
    """Piecewise function for sequential linear segments.

    arguments: Segwise(bounding_symbol, (equation, upper_bound), (equation_2, upper_bound_2))"""
    nargs=None
    @classmethod
    def __new__(cls, *args, **kwargs):
        try:
            # check continuity
            x = args[1]
            if not isinstance(x,sym.Symbol):
                raise ValueError("Segwise's first argument must be a symbol")
            equations = args[2:]
            if len(equations)<2:
                raise ValueError("Segwise requires at least 2 segments to do anything")

            try:
                for i,(eq,ub) in enumerate(equations):
                    if not isinstance(eq,sym.Basic):
                        raise ValueError(f"Segment {i} does not have a valid equation")
                    if isinstance(ub,sym.Basic) and not ub.is_Number:
                        raise ValueError(f"Segment {i} does not have a constant upper bound")
            except TypeError:
                raise ValueError("One or more segments are in an incorrect format")
            except ValueError as v:
                if "too many values to unpack (expected 2)" in v.args:
                    raise ValueError("One or more segments are in an incorrect format")
                raise v

            for i,(eq,ub) in enumerate(equations[:-1]):
                s1 = eq.subs(x,ub)
                s2 = equations[i+1][0].subs(x,ub)
                d1 = sym.diff(eq,x).subs(x,ub)
                d2 = sym.diff(equations[i + 1][0],x).subs(x, ub)
                if not s1.is_Number:
                    raise ValueError(f"Segment {i} contains other variables other than {x}")
                if not s2.is_Number:
                    raise ValueError(f"Segment {i+1} contains other variables other than {x}")
                if not math.isclose(s1,s2,abs_tol=1e-9):
                    raise ValueError(f"Segments {i} and {i+1} are not continuous.")
                if not math.isclose(d1,d2,abs_tol=1e-9):
                    raise ValueError(f"Segments {i} and {i+1} do not have continuous 1st derivatives.")
                if ub>equations[i+1][1]:
                    raise ValueError(f"Segment {i} has a higher upper bound than segment {i+1}")
        except IndexError:
            raise ValueError("Segwise created with zero arguments")

        return super().__new__(*args,**kwargs)
    def _eval_subs(self, old, new):
        if new.is_Number:
            for eq,ub in self.args[1:]:
                if ub>=new:
                    return eq.subs(old,new)
        return super()._eval_subs(old,new)
    def _eval_derivative(self, s):
        if s!=self.args[0]:
            return 0
        return Segwise(s,*((sym.diff(eq,s),ub) for eq,ub in self.args[1:]))



def cubic_spline(x,x_data,y_data):
    """Create a cubic spline"""
    spline = CubicSpline(x_data, y_data, bc_type="natural")
    return Segwise(x,*[(sum(spline.c[m, i] * (x - px)**(3-m) for m in range(4)),spline.x[i+1]) for i,px in enumerate(spline.x[:-1])])