"""
======================
2. Simple Models
======================
"""

# %%
# Here is an overview of the model templates built in to Condor and their mathematical
# representations.

import condor
from condor import operators as ops

# %%
# :class:`ExplicitSystem`
# =======================
#
# An :class:`ExplicitSystem` represents a collection of functions:
#
# .. math::
#    \begin{align}
#    y_{1}&=& f_1\left( x_1, x_2, \ldots, x_n\right) \\
#    y_{2}&=& f_2\left( x_1, x_2, \ldots, x_n\right) \\
#         &\vdots& \\
#    y_{m}&=& f_m\left( x_1, x_2, \ldots, x_n\right) \\
#    \end{align}
#
#
# where each :math:`y_i` is a name-assigned expression on the ``output`` field and each
# :math:`x_i` is an element drawn from the ``input`` field.
# 
# Each :math:`x_i` and :math:`y_j` may have arbitrary shape. Condor can automatically
# calculate the derivatives :math:`\frac{dy_j}{dx_i}` as needed for parent solvers, etc.


class ExplicitShapeDemo(condor.ExplicitSystem):
    x1 = input()
    x2 = input(shape=(2, 3))

    output.y1 = x1 + x2
    output.y2 = 2 * x2


exp_out = ExplicitShapeDemo(x1=1, x2=[[1, 2, 3], [4, 5, 6]])
print(exp_out.output)

# %%
# We have found it useful to embed a collection of models into a single
# ``ExplicitSystem`` to ensure proper binding of the embedded models and evaluate the
# model separately from the evaluation of any parent solvers.


# %%
# :class:`Tablelookup`
# ============================
#
# It is often useful to interpolate pre-existing data. For this, the
# :class:`Tablelookup` model template provides a convenient way to specify the
# interpolant ``input`` and ``output``, which interpolates data on the ``input_data``
# and ``output_data`` fields. This model uses the `ndsplines` library to perform the
# interpolation and compute derivatives as needed. Note that this table model assumes
# fixed input and output data, but a model with variable input and output data could be
# defined as needs arise.
#
# A :class:`TableLookup` is similar to an explicit system, where the functions mapping
# inputs to outputs are tensor product B-splines constructed from a data set on a grid.
#
# Condor supports any number of input dimensions, and provides derivatives
# :math:`\frac{dy_i}{dx_j}` as needed.

import numpy as np
class TableExample(condor.Tablelookup):
    x = input()
    y = output()

    input_data[x] = np.linspace(-1, 1, 100)*np.pi
    output_data[y] = ops.sin(input_data[x])


out = TableExample(np.pi/2)
print(out.y)
assert np.isclose(out.y, 1)


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
# Options
# ----------
#
# Solver options can be passed from throught he Options attribute. Ultimately it is the
# implementations job to parse the Options, but except where different solvers for the
# same model-type conflict the intention is to make the argument manipulation at the
# implementation layer as thin as possible.
#
# In the case of the Tablelookup




