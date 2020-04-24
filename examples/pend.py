import numpy as np
import sympy as sym
import sympy.physics.mechanics as me

import pycollo

final_time = 1

# REMEMBER TO HANDLE NO SUBSTITUTIONS

# Symbols
y0, y1, u0 = sym.symbols('y0 y1 u0')
g = sym.symbols('g')
m0, p0, d0, l0, k0, I0 = sym.symbols('m0 p0 d0 l0 k0 I0')
T0min, T0max = sym.symbols('T0min T0max')

# Optimal Control Problem
problem = pycollo.OptimalControlProblem(
	name="Pendulum swing up problem")
phase = problem.new_phase(name="A")

phase.state_variables = [y0, y1]
phase.control_variables = [u0]

# State equations
phase.state_equations = [y1, (g*m0*p0*sym.cos(y0) + (((T0min + T0max)/2) + u0*(T0max - T0min)))/(m0*(k0**2 + p0**2))]

# Integrand functions
phase.integrand_functions = [(u0**2)]

# Objective function
problem.objective_function = phase.integral_variables[0]

# Point constraints
phase.bounds.initial_state_constraints = {
	y0: -np.pi/2,
	y1: 0,
}
phase.bounds.final_state_constraints = {
	y0: np.pi/2,
	y1: 0,
}

# Bounds
phase.bounds.initial_time = 0
phase.bounds.final_time = final_time
phase.bounds.state_variables = {
	y0: [-np.pi, np.pi],
	y1: [-10, 10],
}
phase.bounds.control_variables = {
	u0: [-0.5, 0.5],
}
phase.bounds.integral_variables = [[0, 1000]]

phase.auxiliary_data = {}

# Guess
phase.guess.time = np.array([0, 1])
phase.guess.state_variables = np.array([[-np.pi/2, np.pi/2], [0, 0]])
phase.guess.control_variables = np.array([[0, 0]])
phase.guess.integral_variables = np.array([100])

# Auxiliary data
problem.auxiliary_data = {g: -9.81, d0: 0.5, k0: 1/12, T0min: -30, T0max: 30,
	m0: 1.0, p0: 1.0,
	}


problem.settings.max_mesh_iterations = 10
problem.settings.display_mesh_result_graph = True
problem.settings.nlp_tolerance = 10e-7
problem.settings.mesh_tolerance = 10e-6
problem.settings.maximise_objective = False
problem.settings.backend = "pycollo"
problem.settings.scaling_method = "none"
problem.settings.assume_inf_bounds = False
problem.settings.inf_value = 1e16
problem.settings.check_nlp_functions = False
problem.settings.collocation_points_min = 2
# problem.settings.collocation_points_max = 8
problem.settings.derivative_level = 1
# phase.mesh.number_mesh_sections = 2
# phase.mesh.number_mesh_section_nodes = [2, 3]
# phase.mesh.mesh_section_sizes = [1/2, 1/2]


# Solve
problem.initialise()
problem.solve()
