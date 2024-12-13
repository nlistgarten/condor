"""
=========================
Sellar's Problem
=========================
"""
# %%
# We wanted to have an API would loook as much like the mathematical description as
# possible with as little distraction from programming cruft as possible.  For example,
# an arbitrary system of equations like from Sellar,
#
# .. math::
#    \begin{align}
#    y_{1}&=x_{0}^{2}+x_{1}+x_{2}-0.2\,y_{2} \\
#    y_{2}&=\sqrt{y_{1}}+x_{0}+x_{1}
#    \end{align}
#
# should be writable as
#
# .. code-block:: python
#
#      y1 == x[0] ** 2 + x[1] + x[2] - 0.2 * y2
#      y2 == y1**0.5 + x[0] + x[1]
#
# Of course, in both the mathematic and programmatic description, the source of each
# symbol must be defined. In an engineering memo, we might say "where :math:`y_1,y_2`
# are the variables to solve and :math:`x \in \mathbb{R}^3` parameterizes the system of
# equations," which suggests the API for an algebraic system of equations as

import condor as co

class Coupling(co.AlgebraicSystem):
    x = parameter(shape=3)
    y1 = variable(initializer=1.)
    y2 = variable(initializer=1.)

    residual(y1 == x[0] ** 2 + x[1] + x[2] - 0.2 * y2)
    residual(y2 == y1**0.5 + x[0] + x[1])

# %%
# which can be evaluated by instantiating the model with numerical values for hte
# parameter, which *binds* the result from the iterative solver to the named *element* and
# *field* attributes on the instance,

coupling = Coupling([5., 2., 1]) # evaluate the model numerically
print(coupling.y1, coupling.y2) # individual elements are bound numerically
print(coupling.variable) # fields are bound as a dataclass


# %%
# Models can be used recursively, building up more sophisticated models by *embedding*
# models within another. However, system encapsolation is enforced so only elements from input and
# output fields are accessible after the model has been defined. For example, we may
# wish to optimize Sellar's algebraic system of equations. Mathematically, we can define
# the optimization as
#
# .. math::
#    \begin{aligned}
#    \operatorname*{minimize}_{x \in \mathbb{R}^3} &  &  & x_{2}^{2}+x_{1}+y_{1}+e^{-y_{2}} \\
#    \text{subject to} &  &  & 3.16\le y_{1}\\
#     &  &  & y_{2}\le24.0
#    \end{aligned}
#
# where :math:`y_1` and :math:`y_2` are the solution to the system of algebraic
# equations described above. In condor, we can write this as

from condor import operators as ops

class Sellar(co.OptimizationProblem):
    x = variable(shape=3, lower_bound=0, upper_bound=10)
    coupling = Coupling(x)
    y1, y2 = coupling

    objective = x[2]**2 + x[1] + y1 + ops.exp(-y2)
    constraint(y1 > 3.16)
    constraint(24. > y2)

# %%
# As with the system of algebraic equations, we can numerically solve this optimization
# problem by providing an initial value for the variables and instantiating the model.
# The resulting object will have a dot-able data structure with the bound results,
# including the embedded ``Coupling`` model:

Sellar.set_initial(x=[5,2,1])
sellar = Sellar()
print()
print("objective value:", sellar.objective) # scalar value
print(sellar.constraint) # field
print(sellar.coupling.y1) # embedded-model element
