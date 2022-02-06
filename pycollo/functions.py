import sympy as sym
from scipy.interpolate import CubicSpline, PPoly
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
class _PPolyStash(object):
    """hides ppoly instances from sympy"""
    def __init__(self):
        self._cache = {}
        self._next = 0
    def register_poly(self,ppoly:PPoly)->int:
        self._cache[self._next] = ppoly
        self._next+=1
        return self._next-1
    def __getitem__(self, item):
        return self._cache[item]
class PolynomialSpline(sym.Function):
    """Increased performance spline via "data smuggling"?"""
    nargs=None
    _equispaced = True
    poly_cache = _PPolyStash()
    def set_ppoly(self,ppoly:PPoly):
        self._ppoly = ppoly
        equispace = ppoly.x[1]-ppoly.x[0]
        for x1,x2 in zip(ppoly.x[:-1],ppoly.x[1:]):
            if not math.isclose((x2-x1),equispace):
                self._equispaced = False
                break
        else:
            self._equispaced = True
    def _eval_subs(self, old, new):
        sub_arg = self.args[0].subs(old,new)
        if sub_arg.is_Number:
            return self._ppoly(sub_arg)
        new = self.__class__(sub_arg)
        new.set_ppoly(self._ppoly)
        return new
    def _eval_derivative(self, s):
        derivative = self.__class__(*self.args)
        derivative.set_ppoly(self._ppoly.derivative())
        return sym.diff(self.args[0],s)*derivative
    #readonly property
    @property
    def equispaced(self):
        return self._equispaced
    @property
    def _ppoly(self):
        if len(self.args)==1:
            raise RuntimeError("PPoly not set yet!")
        return self.poly_cache[self.args[1]]
    @_ppoly.setter
    def _ppoly(self,ppoly:PPoly):
        idx = self.poly_cache.register_poly(ppoly)
        self._args = (self.args[0],sym.sympify(idx))
class CyclicPolynomialSpline(PolynomialSpline):
    _wrap = None
    def set_ppoly(self,ppoly:PPoly):
        if not math.isclose(ppoly(ppoly.x[0]),ppoly(ppoly.x[-1])):
            raise ValueError("Cyclic spline doesn't have cyclic PPoly!")
        self._wrap = ppoly.x[-1]-ppoly.x[0]
        super().set_ppoly(ppoly)
    def _eval_subs(self, old, new):
        sub_arg = self.args[0].subs(old,new)
        if sub_arg.is_Number:
            return self._ppoly(sub_arg % self._wrap)
        new = self.__class__(sub_arg)
        new.set_ppoly(self._ppoly)
        return new
def cubic_spline(x, x_data, y_data, bounds: typing.Literal["natural", "not-a-knot", "periodic", "clamped"] = "natural"):
    """Create a cubic spline

    Setting periodic bounds will return a CyclicSegwise function"""
    if bounds=="periodic" and not math.isclose(y_data[0],y_data[-1]):
        raise ValueError("Periodic splines require the first and last y data points to be the same.")
    # normalise spline to make 0 the first data point
    min_data = min(x_data)
    spline = CubicSpline([xd - min_data for xd in x_data], y_data, bc_type=bounds)
    pspline = (CyclicPolynomialSpline if bounds=="periodic" else PolynomialSpline)(x-min_data)
    pspline.set_ppoly(spline)
    return pspline

def softplus(x,k=1.0):
    """Approximates x if x>0 else 0 safely"""
    return Segwise(x,(1/k*sym.ln(1+sym.exp(Segwise.s*k)),0),(Segwise.s+1/k*sym.ln(1+sym.exp(-Segwise.s*k)),sym.oo))

def logistic(x,k=1.0):
    """Approximates 1 if x>0 else 0 safely"""
    return 0.5*(sym.tanh(k*x)+1)