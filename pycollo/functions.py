import sympy as sym
from scipy.interpolate import CubicSpline
import math
import typing

class Segwise(sym.Function):
    """Piecewise function for sequential linear segments.

    arguments: Segwise(argument, (equation (in Segwise.s), upper_bound), (equation_2, upper_bound_2))"""
    nargs=None
    s = sym.Symbol("s")
    _equispaced = True
    @classmethod
    def __new__(cls, *args, **kwargs):
        equispaced = True
        try:
            x = args[1]
            equations = args[2:]
        except IndexError:
            raise ValueError("Segwise created with zero arguments")

        if not isinstance(x, sym.Basic):
            raise ValueError("Segwise's first argument must a sympy expression")
        if len(equations)<2:
            raise ValueError("Segwise requires at least 2 segments to do anything")
        try:
            for i,(eq,ub) in enumerate(equations):
                if not isinstance(eq,sym.Basic):
                    raise ValueError(f"Segment {i} does not have a valid equation")
                if isinstance(ub,sym.Basic) and not ub.is_Number:
                    raise ValueError(f"Segment {i} does not have a constant upper bound")
                if ub==sym.oo:
                    #can't do floor division with infinity...
                    equispaced = False
        except TypeError:
            raise ValueError("One or more segments are in an incorrect format")
        except ValueError as v:
            if "too many values to unpack (expected 2)" in v.args:
                raise ValueError("One or more segments are in an incorrect format")
            raise v

        segment_spacing = equations[1][1]-equations[0][1]
        # check upper bounds
        for i,(eq,ub) in enumerate(equations[:-1]):
            if ub>equations[i+1][1]:
                raise ValueError(f"Segment {i} has a higher upper bound than segment {i+1}")
            if not math.isclose(equations[i+1][1]-ub,segment_spacing,abs_tol=1e-9):
                equispaced=False

        obj = super().__new__(*args,**kwargs)
        obj._equispaced = equispaced
        return obj
    def check_continuity(self):
        """Check that this Segwise instance is continuous with continuous 1st derivatives
        May be slow for large numbers of segments"""
        equations = self.args[1:]
        # check continuity
        for i,(eq,ub) in enumerate(equations[:-1]):
            s1 = eq.subs(self.s,ub).evalf()
            s2 = equations[i+1][0].subs(self.s,ub).evalf()
            d1 = sym.diff(eq,self.s).subs(self.s,ub).evalf()
            d2 = sym.diff(equations[i + 1][0],self.s).subs(self.s, ub).evalf()
            if not s1.is_Number:
                print(f"Segment {i} contains other variables other than {self.s}")
                return False
            if not s2.is_Number:
                print(f"Segment {i+1} contains other variables other than {self.s}")
                return False
            if not math.isclose(s1,s2,abs_tol=1e-9):
                print(f"Segments {i} and {i+1} are not continuous.")
                return False
            if not math.isclose(d1,d2,abs_tol=1e-9):
                print(f"Segments {i} and {i+1} do not have continuous 1st derivatives.")
                return False
        return True
    def _eval_subs(self, old, new):
        sub_arg = self.args[0].subs(old,new)
        if sub_arg.is_Number:
            for eq,ub in self.args[1:]:
                if ub>=sub_arg:
                    return eq.subs(self.s,sub_arg)
        return Segwise(sub_arg,*self.args[1:])
    def _eval_derivative(self, s):
        return sym.diff(self.args[0],s)*self.__class__(self.args[0],*((sym.diff(eq,self.s),ub) for eq,ub in self.args[1:]))
    #readonly property
    @property
    def equispaced(self):
        return self._equispaced

class CyclicSegwise(Segwise):
    """Piecewise function for repeating sequential linear segments.

    Evaluates as Segwise(argument % highest_upper_bound)

    arguments: Segwise(argument, (equation (in Segwise.s), upper_bound), (equation_2, upper_bound_2))"""
    def check_continuity(self):
        """Check that this Segwise instance is continuous with continuous 1st derivatives
        May be slow for large numbers of segments"""
        if super().check_continuity():
            equations = self.args[1:]
            # check continuity
            first, fub = equations[0]
            last, lub = equations[-1]
            s1 = first.subs(self.s,0)
            s2 = last.subs(self.s,lub)
            d1 = sym.diff(first,self.s).subs(self.s,0)
            d2 = sym.diff(last,self.s).subs(self.s, lub)
            if not s1.is_Number:
                print(f"The first segment contains other variables other than {self.s}")
                return False
            if not s2.is_Number:
                print(f"The last segment contains other variables other than {self.s}")
                return False
            if not math.isclose(s1,s2,abs_tol=1e-9):
                print(f"The first and last segment are not continuous.")
                return False
            if not math.isclose(d1,d2,abs_tol=1e-9):
                print(f"The first and last segment do not have continuous 1st derivatives.")
                return False
            return True
        return False
    def _eval_subs(self, old, new):
        sub_arg = self.args[0].subs(old,new)
        if sub_arg.is_Number:
            sub_arg = sub_arg % self._wrap
            for eq,ub in self.args[1:]:
                if ub>=sub_arg:
                    return eq.subs(self.s,sub_arg)
        return CyclicSegwise(sub_arg,*self.args[1:])
    #readonly property
    @property
    def equispaced(self):
        return self._equispaced
    @property
    def _wrap(self):
        return self.args[-1][1]


def cubic_spline(x, x_data, y_data, bounds: typing.Literal["natural", "not-a-knot", "periodic", "clamped"] = "natural"):
    """Create a cubic spline

    Setting periodic bounds will return a CyclicSegwise function"""
    if bounds=="periodic" and not math.isclose(y_data[0],y_data[-1]):
        raise ValueError("Periodic splines require the first and last y data points to be the same.")
    # normalise spline to make 0 the first data point
    min_data = min(x_data)
    spline = CubicSpline([xd - min_data for xd in x_data], y_data, bc_type=bounds)
    return (CyclicSegwise if bounds == "periodic" else Segwise)(x - min_data,*[(sum(spline.c[m, i] * (Segwise.s - px)**(3-m) for m in range(4)),spline.x[i+1]) for i,px in enumerate(spline.x[:-1])])

def softplus(x,k=1.0):
    """Approximates x if x>0 else 0 safely"""
    return Segwise(x,(1/k*sym.ln(1+sym.exp(Segwise.s*k)),0),(Segwise.s+1/k*sym.ln(1+sym.exp(-Segwise.s*k)),sym.oo))

def logistic(x,k=1.0):
    """Approximates 1 if x>0 else 0 safely"""
    return 0.5*(sym.tanh(k*x)+1)