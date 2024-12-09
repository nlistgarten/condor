"""
=========================
3. Optimization Problem
=========================
"""

import condor
from condor import operators as ops

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

# .. math::
#
#    \begin{aligned}
#    \operatorname*{minimize}_{x_{1}, \ldots, x_{n}} &  &  &
#         f (x_1, \ldots, x_n, p_1, \ldots p_m ) \\
#    \text{subject to} &  &  & l_{x_i} \le x_i \le u_{x_i} \\
#    & & & l_{g_i} \le g_i (x_1, \ldots , x_n, p_1, \ldots p_m ) \le u_{g_i} \\
#    \end{aligned}
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
