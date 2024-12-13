"""
=========================
1. Introduction to Condor
=========================
"""
# %%
# Condor is *modeling framework* with a *Domain Specific Language* (DSL) for
# constructing mathematical models.
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
# Condor uses metaprogramming to to turn the class *declaration* mechanism into a
# blackboard-like environment to achieve the desired API. This approach helps us see
# these mathematical models as datastructures that can then be transformed as needed to
# automate the process that is typically performed manually for defining and evaluating
# mathematical models in engineering analysis,
#
# .. figure:: /images/math-model-process.png
#    :width: 100%
#
# We followed modern pythonic best-practices and patterns to settle on a multi-layered
# architecture like the Model-View-Controller paradigm in web development. The
# three key components of the architecture are:
#
# - The model layer, which provides an API for users to write their model. Condor models
#   are ultimately a data structure which represents the represents the user's
#   mathematical intent for the model.
# - The backend layer provides a consistent interface to a third party *Computational
#   Engine*, a symbolic-computational library which provides symbolic representation of
#   *elements* and *operations* with awareness for basic differential calculus. The goal
#   for the backend is provide a thin wrapper with a consistent interface so the
#   computational engine implementation could be swapped out. Currently, we ship with
#   CasADi as the only engine, although we hope to demonstrate a backend module for an
#   alternate backend in the future.
# - The implementation layer is the glue code that operates on the model data structure,
#   using the backend to form the numerical functions needed to call the third-party
#   solvers which implement the nuemrical algorithms of interest. The implementation
#   layer then calls the solver and binds the results to the model instance.
#
# .. figure:: /images/architecture.png
#    :width: 50%

# %%
# ..
#   The Model Layer
#   ================
#
# Each user model is declared as a subclass of a *Model Template*, a ``class`` with a
# ``ModelType`` metaclass, which defines the *fields* from which *elements* are drawn to
# define the model. Condor currently ships with 5 model templates:
#
# +---------------------------+---------------+-----------------------+----------------------+
# |                           |         fields                                               |
# |                           +---------------+-----------------------+----------------------+
# | built-in template         | input         | internal              | output               |
# +===========================+===============+=======================+======================+
# | ``ExplicitSystem``        | - input       |                       | - output             |
# +---------------------------+---------------+-----------------------+----------------------+
# | ``TableLookup``           | - input       | - input_data          | - output             |
# |                           |               | - output_data         |                      |
# +---------------------------+---------------+-----------------------+----------------------+
# | ``AlgebraicSystem``       | - parameter   | - residual            | - variable           |
# |                           |               |                       | - output             |
# +---------------------------+---------------+-----------------------+----------------------+
# | ``TrajectoryAnalysis``    | - parameter   | - state               | - trajectory_output  |
# |                           |               | - modal.action        |                      |
# +---------------------------+---------------+-----------------------+----------------------+
# | ``OptimizationProblem``   | - parameter   | - objective           | - variable           |
# |                           |               | - constraint          |                      |
# +---------------------------+---------------+-----------------------+----------------------+
#
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

# %%
# As another example, if we were interested in transforming Cartesian coordinates to polar form:
#
# .. math::
#    \begin{align}
#    p_r &= \sqrt{x^2 + y^2} \\
#    p_{\theta} &= tan^{-1}\left(\frac{y}{x}\right)
#    \end{align}
#
# We can implement this with an ``ExplicitSystem`` by declaring the inputs and outputs
# of this system as follows:

class PolarTransform(co.ExplicitSystem):
    x = input()
    y = input()

    output.r = ops.sqrt(x**2 + y**2)
    output.theta = ops.atan2(y, x)


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
#    r &= p_r (x^*, y^*) \\
#    \theta &= p_{\theta} (x^*, y^*)
#
# and letting an iterative solver find the solution :math:`x^*,y^*` satisfying both
# residual equations given parameters :math:`r` and :math:`\theta`. In Condor,

class CartesianTransform(co.AlgebraicSystem):
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


out = CartesianTransform(r=1, theta=ops.pi / 4)
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
