"""Work out the optimum mass for maximum cannonball range"""
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import atmosphere

import pycollo
from pycollo.functions import cubic_spline

# state variables
r = sym.Symbol("r") # downrange distance
h = sym.Symbol("h") # height (above sea level?)
v = sym.Symbol("v") # velocity
y = sym.Symbol("y") # velocity angle
# state parameter
radius = sym.Symbol("rad")

#atmospheric density spline
rho = cubic_spline(h,atmosphere.altitudes,atmosphere.rho_data)
#constants
g = 9.81
density = 7870
Cd = 0.5
cannon_energy = 400000
# cannonball parameters
m = 4/3*np.pi*radius**3*density
sa = np.pi*radius**2
#drag
D = 0.5*rho*v**2*sa*Cd

state_equations = {
    r: v*sym.cos(y),
    h: v*sym.sin(y),
    v: -D/m-g*sym.sin(y),
    y: -g*sym.cos(y)/v
}
problem = pycollo.OptimalControlProblem("Optimising Cannonball Radius",parameter_variables=[radius])
phase = problem.new_phase("parabola",[r,h,v,y])
phase.state_equations = state_equations
problem.objective_function = -phase.final_state_variables[0]
phase.bounds.initial_time = 0.0
phase.bounds.final_time = [1,3600] # unlikely to take an hour to land
phase.bounds.initial_state_constraints = {
    r: 0.0,
    h: 0.0,
}
phase.bounds.state_variables = {
    r: [0,1e6],
    h: [0,np.max(atmosphere.altitudes)],
    v: [1,1e6],
    y: [-np.pi/2,np.pi/2]
}
phase.bounds.final_state_constraints = {
    h: 0,
}
phase.path_constraints = [1/2*m*v**2]
phase.bounds.path_constraints = [[0,cannon_energy]]
problem.bounds.parameter_variables = {radius:[0,10]} # 20 metre diameter cannon ball is unlikely to go very far
problem.guess.parameter_variables = [0.05]
phase.guess.time = [0, 60]
phase.guess.state_variables = [[0, 1000], [0, 0], [1,1], [0,0]]
problem.settings.max_mesh_iterations=5
problem.initialise()
problem.solve()
optimal_radius = problem.solution.parameter[0]
print(f"""Cannonball radius: {optimal_radius} m
Cannonball mass: {m.subs(radius,optimal_radius)} kg
Launch angle: {np.rad2deg(problem.solution.state[0][3][0])} degrees
Maximum range: {problem.solution.state[0][0][-1]} m""")
plt.plot(problem.solution.state[0][0],problem.solution.state[0][1])
plt.ylabel("Altitude")
plt.xlabel("Range")
plt.show()