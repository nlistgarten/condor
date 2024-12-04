"""
=========================
1. Introduction to Condor
=========================
"""


# %%
# Condor is *modeling framework* with a *Domain Specific Language* (DSL) for
# constructing mathematical models. The DSL is constructed by using metaprogramming to
# change the behavior of class construction to create blackboard-like where users can
# focus on the mathematical and engineering analysis and not get distracted by the
# programming cruft.
#
# For example, if we were interested in transforming Cartesian coordinates to polar form:
#
# .. math::
#    \begin{align}
#    p_r &= \sqrt{x^2 + y^2} \\
#    p_{\theta} &= tan^{-1}\left(\frac{y}{x}\right)
#    \end{align}
#
# We can implement this with an ``ExplicitSystem`` by declaring the inputs and outputs
# of this system as follows:

import condor
import numpy as np

class PolarTransform(condor.ExplicitSystem):
    x = input()
    y = input()

    output.r = np.sqrt(x**2 + y**2)
    output.theta = np.atan2(y, x)


# %%
# In general, once you've defined any system in Condor, you can just evaulate it
# numerically by passing in numbers:

p = PolarTransform(x=3, y=4)
print(p)

# %%
# The output returned by such a call is designed for inspection to the extent that we
# recommend working in an interactive session or debugger, especially when getting
# accustomed to Condor features.
#
# For example, the outputs of an explicit system are accessible directly:

print(p.r)

# %%
# They can also be retrieved collectively:

print(p.output)

# %%
# You can of course call it again with different arguments

print(PolarTransform(x=1, y=0).output.asdict())


# %%
# While the *binding* of the results in a datastructure is nice, the real benefit of
# constructing condor models is in calling iterative solvers. For example, we could
# perform symbolic manipulation to define  another ``ExplicitSystem`` with :math:`x =
# r\cos\theta` and :math:`y = r\sin\theta`. Or we can we use Condor to
# numerically solve this algebraic system of equations using an ``AlgebraicSystem`` by
# declaring the input radius and angle as ``parameter``\s and the solving variables for
# :math:`x` and :math:`y`. Mathematically, we are defining the system of algebraic
# equations 
#
# .. math::
#    r &= p_r (x^*, y^*) =  \sqrt{x^{*^2} + y^{*^2}} \\
#    \theta &= p_{\theta} (x^*, y^*) = \tan^{-1}\left(\frac{y^*}{x^*}\right)
#
# and letting an iterative solver find the solution :math:`x^*,y^*` satisfying both
# residual equations given parameters :math:`r` and :math:`\theta`. In Condor,

class CartesianTransform(condor.AlgebraicSystem):
    # r and theta are input parameters
    r = parameter()
    theta = parameter()

    # solver will vary x and y to satisfy the residuals
    x = variable(initializer=1)
    y = variable(initializer=1)

    # get r, theta from solver's x, y
    p = PolarTransform(x=x, y=y)

    # residuals to converge to 0
    residual(r == p.r)
    residual(theta == p.theta)


out = CartesianTransform(r=1, theta=np.pi / 4)
print(out.x, out.y)

# %%
# Note also that passing the inputs (or any intermediates) to plain numeric functions
# that can handle symbolic objects as well as pure numerical objects (float or numpy
# arrays) could work for this simple example. However, since we *embedded* the
# ``PolarTransform`` model in this solver, the system evaluated with the solved variable
# values is directly accessible if the ``bind_embedded_models`` option is ``True``
# (which it is by default), as in:


print(out.p.output)


# %%
# Condor ships with a number of useful model types, with a particular emphasis on
# ``OptimizationProblem`` and ``TrajectoryAnalysis`` to support the vehicle analysis and
# design work that motivated the development of Condor.
