"""
============================
:class:`TableLookup` Basics
============================
"""
# %%
#
# It is often useful to interpolate pre-existing data. For this, the
# :class:`TableLookup` model template provides a convenient way to specify the
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

import condor
from condor import operators as ops
import numpy as np
class SinTable(condor.TableLookup):
    x = input()
    y = output()

    input_data[x] = np.linspace(-1, 1, 5)*np.pi
    output_data[y] = ops.sin(input_data[x])


out = SinTable(np.pi/2)
print(out.y)
assert np.isclose(out.y, 1)


class Table(condor.TableLookup):
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
# :class:`TableLookup.Options`
# ------------------------------
#
# Solver options can be passed from throught he Options attribute. Ultimately it is the
# implementations job to parse the Options, but except where different solvers for the
# same model-type conflict the intention is to make the argument manipulation at the
# implementation layer as thin as possible.
#
# In the case of the TableLookup, Options can be used to specify the boundary conditions
# and interpolant degree (in each direction.) Options can be declared during the model
# declaration, as in:

class SinTable(condor.TableLookup):
    x = input()
    y = output()

    input_data[x] = np.linspace(-1, 1, 5)*np.pi
    output_data[y] = ops.sin(input_data[x])

    class Options:
        degrees = 0

print(SinTable(np.pi/4))


# %%
# or by assigning an attribute directly on the Model's Option attribute, which will be
# injected if it is not declared. For example, we can iterate over piecewise constant,
# piecewise linear, and piecewise cubic polynomials to 

class SinTable(condor.TableLookup):
    x = input()
    y = output()

    input_data[x] = np.linspace(-1, 1, 5)*np.pi
    output_data[y] = ops.sin(input_data[x])

    # note, ``Options`` attribute will get created so it can be modified in the loop
    # below

eval_x = np.linspace(-1, 1, 100)*np.pi
from matplotlib import pyplot as plt
fig, ax = plt.subplots(constrained_layout=True)
for k in [0, 1, 3]:
    SinTable.Options.degrees = k
    if k == 3:
        # for cubic polynomial, use constant slope (constant first derivative, 0 second
        # derivative) boundary condition instead of default not-a-knot (constant,
        # non-zero, second derivative)
        SinTable.Options.bcs = (2, 0)
    y = [SinTable(x).y for x in eval_x]
    plt.plot(eval_x, y, label=f"k={k}")
plt.plot(SinTable.input_data[SinTable.x], SinTable.output_data[SinTable.y], 'ko')
plt.plot(eval_x, np.sin(eval_x), 'k--', label="true")
plt.grid(True)
plt.legend()
plt.show()

