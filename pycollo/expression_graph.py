import abc
import functools
import itertools
import numbers
from timeit import default_timer as timer
import weakref

from ordered_set import OrderedSet
import numpy as np
import sympy as sym

from .node import Node

"""

Notes:
------
	Todo - add checking for:
	* Auxiliary data
	* Objective function
	* State equations
	* Path constraints
	* Integrand functions
	* State endpoint constraints
	* Endpoint constraints

	Todo - use sympy.matrix.sparse in hSAD
	
"""

class ExpressionGraph:

	def __init__(self, ocp, problem_variables_user, problem_variables, 
		auxiliary_information, objective, constraints):

		self.ocp = ocp

		self._initialise_node_symbol_number_counters()
		self._initialise_node_mappings()

		self._initialise_problem_variable_information(problem_variables, 
			problem_variables_user)

		self._initialise_default_singleton_number_nodes()
		self._initialise_auxiliary_constant_nodes(auxiliary_information)
		self._initialise_auxiliary_intermediate_nodes()

		self._form_time_normalisation_function()
		self._form_objective_function_and_derivatives(objective)
		self._form_constraints_and_derivatives(constraints)
		self._form_lagrangian_and_derivatives()

	def _initialise_node_symbol_number_counters(self):
		self._number_node_num_counter = itertools.count()
		self._constant_node_num_counter = itertools.count()
		self._intermediate_node_num_counter = itertools.count()
	
	def _initialise_node_mappings(self):
		self._variable_nodes = {}
		self._number_nodes = {}
		self._constant_nodes = {}
		self._intermediate_nodes = {}
		self._precomputable_nodes = {}

	def _initialise_problem_variable_information(self, x_vars, x_vars_user):
		self._initialise_problem_variable_attributes(x_vars, x_vars_user)
		self._initialise_problem_variable_nodes()

	def _initialise_problem_variable_attributes(self, x_vars, x_vars_user):
		x_continuous, x_endpoint = x_vars
		x_continuous_user, x_endpoint_user = x_vars_user

		self.problem_variables_continuous = OrderedSet(x_continuous)
		self.problem_variables_endpoint = OrderedSet(x_endpoint)
		self.problem_variables_continuous_user = OrderedSet(x_continuous_user)
		self.problem_variables_endpoint_user = OrderedSet(x_endpoint_user)
		
		self.user_to_pycollo_problem_variables_mapping_in_order = {
			**dict(zip(self.problem_variables_continuous_user, 
				self.problem_variables_continuous)),
			**dict(zip(self.problem_variables_endpoint_user, 
				self.problem_variables_endpoint)),
			}

		self.lagrange_syms = ()

	def _initialise_problem_variable_nodes(self):
		self.time_function_variable_nodes = set()

		self._continuous_variable_nodes = []
		for x_var_user in self.problem_variables_continuous_user:
			node = Node(x_var_user, self)
			self._continuous_variable_nodes.append(node)
			if str(node.symbol)[1] in ('y', 'u'):
				self.time_function_variable_nodes.add(node)

		self._endpoint_variable_nodes = []
		for x_b_var_user in self.problem_variables_endpoint_user:
			node = Node(x_b_var_user, self)
			self._endpoint_variable_nodes.append(node)

	def _initialise_default_singleton_number_nodes(self):
		self._zero_node = Node(0, self)
		self._one_node = Node(1, self)
		two_node = Node(2, self)
		neg_one_node = Node(-1, self)
		half_node = Node(0.5, self)

	def _initialise_auxiliary_constant_nodes(self, aux_info):
		self.user_symbol_to_expression_auxiliary_mapping = {}
		self._user_constants = OrderedSet()
		for key, value in aux_info.items():
			if isinstance(value, numbers.Real):
				self._user_constants.add(key)
				node = Node(key, self, value=value)
			else:
				self.user_symbol_to_expression_auxiliary_mapping[key] = value

	def _initialise_auxiliary_intermediate_nodes(self):
		iterable = self.user_symbol_to_expression_auxiliary_mapping.items()
		for node_symbol, node_expr in iterable:
			_ = Node(node_symbol, self, equation=node_expr)

	def _form_time_normalisation_function(self):
		self._form_function_and_derivative(
			func=self.ocp._STRETCH,
			wrt=None,
			order=0,
			func_abrv='t_norm',
			init_func=True,
			completion_msg='time normalisation'
			)

	def _form_objective_function_and_derivatives(self, objective):
		self._form_function_and_derivative(
			func=objective,
			wrt=self._endpoint_variable_nodes,
			order=1, 
			func_abrv='J',
			init_func=True,
			completion_msg='objective gradient',
			)

	def _form_constraints_and_derivatives(self, constraints):

		def form_continuous(continuous_constraints):
			form_function_and_derivative(func=continuous_constraints,
				wrt=self._continuous_variable_nodes, func_abrv='c',
				completion_msg='Jacobian of the continuous constraints')

		def form_endpoint(endpoint_constraints):
			form_function_and_derivative(func=endpoint_constraints,
				wrt=self._endpoint_variable_nodes, func_abrv='b',
				completion_msg='Jacobian of the endpoint constraints')

		form_function_and_derivative = functools.partial(
			self._form_function_and_derivative, order=1, init_func=True)

		y_eqns, c_cons, q_funcs, y_b_cons, b_cons = constraints
		continuous_constraints = sym.Matrix(y_eqns + c_cons + q_funcs)
		endpoint_constraints = sym.Matrix(y_b_cons + b_cons)
		
		form_continuous(continuous_constraints)
		form_endpoint(endpoint_constraints)

	def _form_lagrangian_and_derivatives(self):

		def form_objective(L_J):
			form_function_and_derivative(func=L_J,
				wrt=self._endpoint_variable_nodes, func_abrv='L_J',
				completion_msg='Hessian of the objective Lagrangian')

		def form_defect(L_zeta):
			form_function_and_derivative(func=L_zeta,
				wrt=self._continuous_variable_nodes, func_abrv='L_zeta',
				completion_msg='Hessian of the defect Lagrangian')

		def form_path(L_gamma):
			form_function_and_derivative(func=L_gamma,
				wrt=self._continuous_variable_nodes, func_abrv='L_gamma',
				completion_msg='Hessian of the path Lagrangian')

		def form_integral(L_rho):
			form_function_and_derivative(func=L_rho,
				wrt=self._continuous_variable_nodes, func_abrv='L_rho',
				completion_msg='Hessian of the integral Lagrangian')

		def form_endpoint(L_beta):
			form_function_and_derivative(func=L_beta,
				wrt=self._endpoint_variable_nodes, func_abrv='L_beta',
				completion_msg='Hessian of the endpoint Lagrangian')

		def make_L_matrices_lower_triangular():

			def make_lower_triangular(mat):
				return sym.Matrix(np.tril(np.array(mat)))

			self.ddL_dxbdxb_J = make_lower_triangular(self.ddL_J_dxbdxb)
			self.ddL_dxdx_zeta = make_lower_triangular(self.ddL_zeta_dxdx)
			self.ddL_dxdx_gamma = make_lower_triangular(self.ddL_gamma_dxdx)
			self.ddL_dxdx_rho = make_lower_triangular(self.ddL_rho_dxdx)
			self.ddL_dxbdxb_beta = make_lower_triangular(self.ddL_beta_dxbdxb)

		form_function_and_derivative = functools.partial(
			self._form_function_and_derivative, order=2, init_func=False)

		sigma = sym.symbols('_sigma')
		self.ocp.sigma = sigma

		L_syms = [sym.symbols(f'_lambda_{n}') for n in range(self.ocp._num_c)]
		self.ocp.lagrange_syms = L_syms

		self.lagrange_syms = tuple([sigma] + L_syms)
		for L_sym in self.lagrange_syms:
			_ = Node(L_sym, self)

		c_defect_slice = self.ocp._c_defect_slice
		c_path_slice = self.ocp._c_path_slice
		c_integral_slice = self.ocp._c_integral_slice
		c_endpoint_slice = self.ocp._c_boundary_slice

		L_objective = sigma*self.J
		
		L_defect_terms = (self.ocp._STRETCH*l*c 
			for l, c in zip(L_syms[c_defect_slice], self.c[c_defect_slice]))
		L_defect = sum(L_defect_terms, sym.sympify(0))

		L_path_terms = (self.ocp._STRETCH*l*c 
			for l, c in zip(L_syms[c_path_slice], self.c[c_path_slice]))
		L_path = sum(L_path_terms, sym.sympify(0))

		L_integral_terms = (self.ocp._STRETCH*l*c 
			for l, c in zip(L_syms[c_integral_slice], self.c[c_integral_slice]))
		L_integral = sum(L_integral_terms, sym.sympify(0))

		L_endpoint_terms = (l*b 
			for l, b in zip(L_syms[c_endpoint_slice], self.b))
		L_endpoint = sum(L_endpoint_terms, sym.sympify(0))

		form_objective(L_objective)
		form_defect(L_defect)
		form_path(L_path)
		form_integral(L_integral)
		form_endpoint(L_endpoint)

		make_L_matrices_lower_triangular()

	def _form_function_and_derivative(self, func, wrt, order, func_abrv, init_func, completion_msg=None):

		def create_derivative_abbreviation(wrt, func_abrv):
			if wrt is self._continuous_variable_nodes:
				wrt_abrv = 'x'
			elif wrt is self._endpoint_variable_nodes:
				wrt_abrv = 'xb'

			if order == 1:
				return f"d{func_abrv}_d{wrt_abrv}"
			elif order == 2:
				return f"dd{func_abrv}_d{wrt_abrv}d{wrt_abrv}"

		def add_to_namespace(self, args, func_abrv):
			zeroth_arg = f"{func_abrv}"
			first_arg = f"{func_abrv}_nodes"
			second_arg = f"{func_abrv}_precomputable"
			third_arg = f"{func_abrv}_dependent_tiers"

			attribute_names = [zeroth_arg, first_arg, second_arg, third_arg]
			for i, attrib_name in enumerate(attribute_names):
				exec_cmd = f"setattr(self, str('{attrib_name}'), args[{i}])"
				exec(exec_cmd)

			return self

		init_args = self._initialise_function(func)
		if init_func is True:
			self = add_to_namespace(self, init_args, func_abrv)

		for _ in range(order):
			deriv = self.hybrid_symbolic_algorithmic_differentiation(
				*init_args, wrt)
			init_args = self._initialise_function(deriv)

		if order > 0:
			deriv_abrv = create_derivative_abbreviation(wrt, func_abrv)
			self = add_to_namespace(self, init_args, deriv_abrv)

		completion_msg = f"Symbolic {completion_msg} calculated."
		print(completion_msg)

	def _initialise_function(self, expr):

		def substitute_function_for_root_symbols(expr):

			def traverse_root_branch(expr, max_tier):
				root_node = Node(expr, self)
				max_tier = max(max_tier, root_node.tier)
				return root_node.symbol, root_node, max_tier

			if isinstance(expr, sym.Expr):
				root_symbol, root_node, max_tier = traverse_root_branch(expr, 0)
				return root_symbol, [root_node], max_tier
			else:
				expr_subbed = []
				expr_nodes = []
				max_tier = 0
				for entry_expr in expr:
					root_symbol, root_node, max_tier = traverse_root_branch(
						entry_expr, max_tier)
					expr_subbed.append(root_symbol)
					expr_nodes.append(root_node)
				return_matrix = sym.Matrix(np.array(expr_subbed).reshape(
					expr.shape))
				return return_matrix, expr_nodes, max_tier

		def separate_precomputable_and_dependent_nodes(expr, nodes):
			precomputable_nodes = set()
			dependent_nodes = set()
			all_nodes = set(nodes)
			for free_symbol in expr.free_symbols:
				node = self.symbols_to_nodes_mapping[free_symbol]
				all_nodes.update(node.dependent_nodes)
			precomputable_nodes = set()
			dependent_nodes = set()
			for node in all_nodes:
				if node.is_precomputable:
					precomputable_nodes.add(node)
				else:
					dependent_nodes.add(node)
			return precomputable_nodes, dependent_nodes

		def sort_dependent_nodes_by_tier(dependent_nodes, max_tier):
			dependent_tiers = {i: [] for i in range(max_tier+1)}
			for node in dependent_nodes:
				dependent_tiers[node.tier].append(node)
			return dependent_tiers

		def check_root_tier_is_exlusively_continuous_or_endpoint(
			dependent_tiers):
			return None

		(expr_subbed, expr_nodes, 
			max_tier) = substitute_function_for_root_symbols(expr)

		(precomputable_nodes, 
			dependent_nodes) = separate_precomputable_and_dependent_nodes(
			expr_subbed, expr_nodes)

		dependent_tiers = sort_dependent_nodes_by_tier(dependent_nodes, 
			max_tier)

		return expr_subbed, expr_nodes, precomputable_nodes, dependent_tiers

	@property
	def variable_nodes(self):
		return tuple(self._variable_nodes.values())

	@property
	def constant_nodes(self):
		return tuple(self._constant_nodes.values())

	@property
	def number_nodes(self):
		return tuple(self._number_nodes.values())

	@property
	def intermediate_nodes(self):
		return tuple(self._intermediate_nodes.values())
	
	@property
	def root_nodes(self):
		return self.variable_nodes + self.constant_nodes + self.number_nodes

	@property
	def nodes(self):
		return self.root_nodes + self.intermediate_nodes

	@property
	def precomputable_nodes(self):
		return tuple(self._precomputable_nodes.values())

	@property
	def symbols_to_nodes_mapping(self):
		return {
			**self._variable_nodes, 
			**self._constant_nodes, 
			**self._number_nodes, 
			**self._intermediate_nodes,
			}

	def hybrid_symbolic_algorithmic_differentiation(self, target_function, 
		function_nodes, precomputable_nodes, dependent_nodes_by_tier, wrt):

		def differentiate(function_nodes, wrt):
			derivative_full = [[function_node.derivative_as_symbol(single_wrt) 
				for single_wrt in wrt] 
				for function_node in function_nodes]
			return sym.Matrix(derivative_full)

		def compute_target_function_derivatives_for_each_tier(
				dependent_nodes_by_tier_collapsed):
			df_de = []
			for node_tier in dependent_nodes_by_tier_collapsed:
				derivative = differentiate(function_nodes, node_tier)
				df_de.append(derivative)
			return df_de

		def compute_delta_matrices_for_each_tier(num_e0, 
				dependent_nodes_by_tier_collapsed):
			delta_matrices = [1]
			for tier_num, dependent_nodes_tier in enumerate(
					dependent_nodes_by_tier_collapsed[1:], 1):
				num_ei = len(dependent_nodes_tier)
				delta_matrix_i = sym.Matrix.zeros(num_ei, num_e0)
				for by_tier_num in range(tier_num):
					delta_matrix_j = delta_matrices[by_tier_num]
					deriv_matrix = differentiate(dependent_nodes_tier, 
						dependent_nodes_by_tier_collapsed[by_tier_num])
					delta_matrix_i += deriv_matrix*delta_matrix_j
				delta_matrices.append(delta_matrix_i)
			return delta_matrices

		def compute_derivative_recursive_hSAD_algorithm():
			num_f = len(function_nodes)
			derivative = sym.Matrix.zeros(num_f, num_e0)
			for df_dei, delta_i in zip(df_de, delta_matrices):
				derivative += df_dei*delta_i
			return derivative

		dependent_nodes_by_tier_collapsed = [wrt]
		for nodes in list(dependent_nodes_by_tier.values())[1:]:
			if nodes:
				dependent_nodes_by_tier_collapsed.append(nodes)

		df_de = compute_target_function_derivatives_for_each_tier(
			dependent_nodes_by_tier_collapsed)

		num_e0 = len(dependent_nodes_by_tier_collapsed[0])
		delta_matrices = compute_delta_matrices_for_each_tier(num_e0, 
			dependent_nodes_by_tier_collapsed)

		derivative = compute_derivative_recursive_hSAD_algorithm()

		return derivative

	def __str__(self):
		cls_name = self.__class__.__name__
		return (f"{cls_name}(({self.problem_variables_continuous_user}, "
			f"{self.problem_variables_endpoint_user}))")

	def __repr__(self):
		cls_name = self.__class__.__name__
		return (f"{cls_name}(problem_variables_user="
			f"({self.problem_variables_continuous_user}, "
			f"{self.problem_variables_endpoint_user}))")







def kill():
		print('\n\n')
		raise ValueError

def cout(*args):
	print('\n\n')
	for arg in args:
		print(f'{arg}\n')


