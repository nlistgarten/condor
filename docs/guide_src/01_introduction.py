"""
=========================
1. Introduction to Condor
=========================
"""

# %%
# To get a sense of working with Condor, we'll start with some concrete examples making
# use of the fundamental building blocks of models and how they are used together.
#
# Condor provides a baseline set of what we call *model templates*, which you might
# think of as representing specific mathematical models. In this way, Condor models
# often look very much like what you'd see in a mathematical description of a model in
# code form.
#
# Explicit Systems
# ================
#
# First, there's the ``ExplicitSystem``, which takes inputs and produces outputs that
# are functions of those inputs. As an example, suppose we want to define a system for
# transforming Cartesian coordinates to polar form:
#
# .. math::
#    \begin{align}
#    r &= \sqrt{x^2 + y^2} \\
#    \theta &= tan^{-1}\left(\frac{y}{x}\right)
#    \end{align}
#
# We can implement this with an ``ExplicitSystem`` as follows:

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

out = PolarTransform(x=3, y=4)
print(out)

# %%
# The output returned by such a call is designed for inspection to the extent that we
# recommend working in an interactive session or debugger, especially when getting
# accustomed to Condor features.
#
# For example, the outputs of an explicit system are accessible directly:

print(out.r)

# %%
# They can also be retrieved collectively:

print(out.output)

# %%
# You can of course call it again with different arguments

print(PolarTransform(x=1, y=0).output.asdict())

# %%
# Nesting Systems
# ---------------
#
# Systems may be nested arbitrarily. This is done by simply calling an existing
# system as above, but within another system definition.


class Hypot(condor.ExplicitSystem):
    x = input()
    y = input()
    output.hypot = np.sqrt(x**2 + y**2)


class PolarTransform(condor.ExplicitSystem):
    x = input()
    y = input()
    # "call" Hypot and grab its output
    output.r = Hypot(x, y).hypot
    output.theta = np.atan2(y, x)


# %%
# Note also that passing the inputs (or any intermediates) to plain numeric functions
# works fine too, however... TODO

# TODO maybe a quick sidebar/note/hint on how this works


# %%
# Algebraic Systems
# =================
#
# With just explicit systems, you can already model quite a few things, but you're not
# gaining much over plain functions besides getting a result object providing "dot
# access" to retrieve outputs by name.
#
# Algebraic systems are used in situations where you need an iterative method to solve a
# system of equations for which there may not be an analytic solution. Algebraic systems
# are expressed in canonical residual form:
#
# .. math::
#    R(x, u) = 0
#
# Where :math:`x` are variables a solver can vary to satisfy the residuals (to some
# tolerance) and :math:`u` are parameters.
#
# Suppose we want an inverse of the ``PolarTransform`` above. Of course, this is
# achievable by defining another ``ExplicitSystem`` with :math:`x = r\cos\theta` and
# :math:`y = r\sin\theta`, but for demonstration let's write it as follows:
#
# .. math::
#    r - \sqrt{x^{*^2} + y^{*^2}} = 0 \\
#    \theta - \tan^{-1}\left(\frac{y^*}{x^*}\right) = 0
#
# The idea here is that instead of solving these coupled equations algebraically, we can
# let an iterative solver find the solution :math:`x^*,y^*` satisfying both residual
# equations given parameters :math:`r` and :math:`\theta`.


class CartesianTransform(condor.AlgebraicSystem):
    # r and theta are input parameters
    r = parameter()
    theta = parameter()

    # solver will vary x and y to satisfy the residuals
    x = implicit_output(initializer=1)
    y = implicit_output(initializer=1)

    # get r, theta from solver's x, y
    p = PolarTransform(x=x, y=y)

    # residuals to converge to 0
    residual.resid_r = r - p.r
    residual.resid_theta = theta - p.theta


out = CartesianTransform(r=1, theta=np.pi / 4)
print(out.x, out.y)
