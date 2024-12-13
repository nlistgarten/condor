"""
================================
3. System of Algebraic Equations
================================
"""

import condor
from condor import operators as ops


# %%
# :class:`AlgebraicSystem`
# ========================
#
# An :class:`AlgebraicSystem` represents a collection of residual functions of
# ``parameter``\s :math:`u` and ``variable``\s :math:`x`, which are driven to a solution
# :math:`x^*`:
#
# .. math::
#    \begin{align}
#    0 &=& R_1(u_1, \dots, u_m, x_1^*, \dots, x_n^*) \\
#    & \vdots & \\
#    0 &=& R_n(u_1, \dots, u_m, x_1^*, \dots, x_n^*)
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
# As a convenience, an additional `output` field can be used to compute additional
# expressions as functions of the paramater and (solved) variable values.


class AlgebraicSys(condor.AlgebraicSystem):
    m = parameter()
    b = parameter()
    y = parameter()
    x = variable()

    residual(m*x + b == y)

    output.r = ops.sqrt(x**2 + y**2)



alg_out = AlgebraicSys(m=0.5, b=1., y = 3.)
print(alg_out.variable)
print(alg_out.output)

# %%
# The :attr:`.AlgebraicSystem.variable` field
# --------------------------------------------
#
# The :attr:`~.AlgebraicSystem.variable` field of :class:`.AlgebraicSystem` and
# :class:`.OptimizationProblem` is an :class:`~condor.fields.InitializedField` which means any element drawn from it has an additional
# `initializer` attribute. This can be set at any time, and will be used as the initial
# value to the iterative solver.
#
# A symbolic expression of the ``parameter`` elements can be used as well. 
#
# By default, when the solver is called and a solution is found, the initializer for
# each variable will be updated to the new solution. This behavior does not apply to
# initialzier's which are symbolic expressions. This behavior can be prevented for
# numerical initializers by setting the ``warm_start`` attribute to ``False``.
#
# Just checking if :attr:`variable` is rendering correctly
#
# Available :class:`Options`
# ------------------------
#
# The key 






