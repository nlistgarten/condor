"""
======================
2. Simple Models
======================
"""

# %%
# Here is an overview of the model templates built in to Condor and their mathematical
# representations.

import condor

# %%
# :class:`ExplicitSystem`
# =======================
#
# An :class:`ExplicitSystem` represents a collection of functions:
#
# .. math::
#    \begin{align}
#    y_1 &=& f_1(x_1, x_2, \dots, x_n) \\
#    y_2 &=& f_2(x_1, x_2, \dots, x_n) \\
#    & \vdots \\
#    y_m &=& f_m(x_1, x_2, \dots, x_n)
#    \end{align}
#
#  where each :math:`y_i` is a name-assigned expression on the ``output`` field and each
#  :math:`x_i` is an element drawn from the ``input`` field.
# 
# Each :math:`x_i` and :math:`y_j` may have arbitrary shape. Condor can automatically
# calculate the derivatives :math:`\frac{dy_j}{dx_i}` as needed for parent solvers, etc.


class ExplicitSys(condor.ExplicitSystem):
    x1 = input()
    x2 = input(shape=(2, 3))

    output.y1 = x1 + x2
    output.y2 = 2 * x2


exp_out = ExplicitSys(x1=1, x2=[[1, 2, 3], [4, 5, 6]])
print(exp_out.output)

# %%
# We have found it useful to embed a collection of models into a isngle ExplicitSystem
# to 


# %%
# :class:`AlgebraicSystem`
# ========================
#
# An :class:`AlgebraicSystem` represents a collection of residual functions of
# parameters :math:`u` and implicit variables :math:`x`, which are driven to a solution
# :math:`x^*`:
#
# .. math::
#    \begin{align}
#    R_1(u_1, \dots, u_m, x_1^*, \dots, x_n^*) &=& 0 \\
#    \vdots & & \\
#    R_n(u_1, \dots, u_m, x_1^*, \dots, x_n^*) &=& 0
#    \end{align}
#
# Condor solves for the :math:`x_i^*` and can automatically calculate the derivatives
# :math:`\frac{dx_i}{du_j}` as needed for parent solvers, etc.


class AlgebraicSys(condor.AlgebraicSystem):
    u = parameter()
    x1 = variable(initializer=1)
    x2 = variable()
    residual(x1**2 == u)
    residual(x1 == x2)


alg_out = AlgebraicSys(4)
print(alg_out.variable)

# %%
# :class:`OptimizationProblem`
# ============================
#
# An :class:`OptimizationProblem` has the general nonlinear program (NLP) form:
#
# .. math::
#    \begin{align}
#        &\underset{x}{\text{minimize}} & f(x) & \\
#        &\text{subject to} & l_x \le x \le u_x & \\
#        &                  & l_g \le g_i(u, x) \le u_g
#    \end{align}
#
# Provided with parameters :math:`u` and initial values for the variables :math:`x`,
# Condor solves for :math:`x^*` to minimize the objective while satisfying constraints.
# Naturally, Condor can provide the derivatives :math:`\frac{dx}{du}` at the solution
# for use as an embedded explicit system under a higher-level solver.


class OptProb(condor.OptimizationProblem):
    u = parameter()
    x = variable(upper_bound=2)

    objective = x**2 + 1 / u

    constraint(2 * x, lower_bound=1)


opt_out = OptProb(u=10)
print(opt_out.x)

# %%
# :class:`TrajectoryAnalysis`
# ===========================
#
# A :class:`TrajectoryAnalysis` solves an initial value problem (IVP) expressed as a
# system of first-order ordinary differential equations (ODEs):
#
# .. math::
#    \begin{align}
#    \dot{x}_i(t) &= f_i(x_1, \dots, x_n, t), & i \in {1, \dots, n} \\
#    y_j &= g_j\left(x_1(t_f), \dots, x_n(t_f)\right), & j \in {1, \dots, m}
#    \end{align}
#
# TODO events
# Condor solves for the states :math:`x_i(t)` and can provided the derivatives of the
# outputs with respect to the initial conditions :math:`\frac{dy_j}{dx_i(0)}` as well as
# the parameters :math:`\frac{dy_j}{du}`

# %%
# :class:`TableLookup`
# ====================
#
# A :class:`TableLookup` is similar to an explicit system, where the functions mapping
# inputs to outputs are tensor product B-splines constructed from a data set on a grid.
#
# Condor supports any number of input dimensions, and provides derivatives
# :math:`\frac{dy_i}{dx_j}` as needed.


class Table(condor.Tablelookup):
    x1 = input()
    x2 = input()

    y1 = output()

    input_data[x1] = [-1, -0.5, 0, 0.5, 1]
    input_data[x2] = [0, 1, 2, 3]

    output_data[y1] = [
        [0, 1, 2, 3],
        [3, 4, 5, 6],
        [6, 7, 8, 9],
        [8, 7, 6, 5],
        [4, 3, 2, 1],
    ]


tab_out = Table(x1=0.5, x2=0.1)
print(tab_out.output)


# %%
# :class:`ExternalSolverWrapper`
# ==============================
#
# External solver systems are not really mathematically distinct from explicit systems.
# They may be thought of as a collection of explicit functions. The distinction is in
# usage -- an external solver system provides a convenient method of wrapping existing
# computational routines and providing derivatives if available.

# TODO maybe we should just forgo the code sample here

import subprocess
import sys


class ExternalSolverWrapper(condor.ExternalSolverWrapper):

    def __init__(self):
        self.input(name="x1")
        self.input(name="x2")
        self.output(name="y1")
        self.output(name="y2")

    def function(self, x1, x2):
        out = subprocess.run(
            [sys.executable, "-c", f"print(2*{x1}); print({x2}**3)"],
            capture_output=True,
            text=True,
        )
        y1, y2 = map(float, out.stdout.split())
        return y1, y2

    def jacobian(self, x1, x2):
        # may provide jacobian dy/dx if available for use within a parent solver
        ...


ExtSys = ExternalSolverWrapper()
ext_out = ExtSys(1, 2)
print(ext_out.output)
