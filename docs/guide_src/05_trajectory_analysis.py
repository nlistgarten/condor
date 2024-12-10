"""
=========================
5. Trajectory Analysis
=========================
"""

import condor
from condor import operators as ops

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

