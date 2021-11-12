"""Minimise force integral against a varying wind"""
import sympy as sym
import numpy as np

import pycollo
from pycollo.functions import cubic_spline

L = 10.0
# state variables
x = sym.Symbol("x")
v = sym.Symbol("v")
#control variable
f = sym.Symbol("f")
f_max = 10
# wind
cda = 0.001
wind_speed = cubic_spline(x,np.linspace(0,L,10),np.random.randn(10))*10

state_equations = {
    x: v,
    v: f-cda*(v-wind_speed)**2
}
problem = pycollo.OptimalControlProblem("Wind resistance - minimum force squared")
phase = problem.new_phase("Stage1",[x,v],f)
phase.state_equations = state_equations
phase.integrand_functions = [f**2]
problem.objective_function = phase.integral_variables[0]
phase.bounds.initial_time = 0.0
phase.bounds.final_time = [0,60]
phase.bounds.initial_state_constraints = {
    x: 0.0,
    v: 0.0
}
phase.bounds.state_variables = {
    x: [0,L],
    v: [0,343],
}
phase.bounds.final_state_constraints = {
    x: L,
    v: 0,
}
phase.bounds.control_variables = {
    f: [-f_max, f_max]
}
phase.guess.time = [0, 10]
phase.guess.state_variables = [[0, L], [0, 0]]
phase.guess.control_variables = [[0, 0]]
phase.bounds.integral_variables = [[0, 100]]
phase.guess.integral_variables = [0]
problem.settings.display_mesh_result_graph = True
problem.initialise()
problem.solve()