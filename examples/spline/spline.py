"""Fly a mass over a hill"""
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

import pycollo
from pycollo.functions import cubic_spline

L = 10.0
# state variables
x = sym.Symbol("x")
vx = sym.Symbol("vx")
y = sym.Symbol("y")
vy = sym.Symbol("vy")
v_max = 343
#control variables
f = sym.Symbol("f")
f_angle = sym.Symbol("f_ang")
fx = f*sym.cos(f_angle)
fy = f*sym.sin(f_angle)
f_max = 10
# terrain
terrain_data = [0,0.2,1,2,1.5,1.2,1.5,2.0,1.8,1.0,0.5,0,0]
spline_hill = cubic_spline(x,np.linspace(0,L,len(terrain_data)),terrain_data)
print(spline_hill)
sine_hill = 1-sym.cos(x/L*sym.pi*2)
hill = spline_hill
g = 9.81

state_equations = {
    x: vx,
    y: vy,
    vx: fx,
    vy: fy-g
}
problem = pycollo.OptimalControlProblem("Flying over a hill")
phase = problem.new_phase("Stage1",[x,y,vx,vy],[f,f_angle])
phase.state_equations = state_equations
problem.objective_function = phase.final_time_variable
phase.bounds.initial_time = 0.0
phase.bounds.final_time = [0,60]
phase.bounds.initial_state_constraints = {
    x: 0.0,
    y: 0.0,
    vx: 0.0,
    vy: 0.0
}
phase.bounds.state_variables = {
    x: [0,L],
    y: [0,1000],
    vx: [-v_max,v_max],
    vy: [-v_max,v_max]
}
phase.bounds.final_state_constraints = {
    x: L,
    y: 0,
    vx: 0,
    vy: 0
}
phase.bounds.control_variables = {
    f: [-f_max, f_max],
    f_angle: [-2*sym.pi,2*sym.pi]
}
phase.path_constraints = [y-hill]
phase.bounds.path_constraints = [[0,0.3]]
phase.guess.time = [0, 60]
phase.guess.state_variables = [[0, L], [0, 0], [0,0], [0,0]]
phase.guess.control_variables = [[0, 0], [0,0]]
problem.settings.max_mesh_iterations=5
problem.initialise()
problem.solve()
plt.scatter(np.linspace(0,L,len(terrain_data)),terrain_data)
x_locs = np.linspace(0,L,100)
plt.plot(x_locs,[hill.subs(x,xl) for xl in x_locs])
plt.plot(problem.solution.state[0][0],problem.solution.state[0][1])
plt.show()