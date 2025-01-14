from dataclasses import dataclass, field

import numpy as np

"""
Need to keep in a separate file so backend.get_symbol_data (required by fields which
generate the symbols) can return filled dataclass without causing a circular import

"""

"""
Backend:
[x] provide symbol_generator for creating backend symbol repr
[x] symbol_class for isinstance(model_attr, backend.symbol_class
[x] name which is how backend options on model is identified
Do we allow more complicated datastructures? like models, etc.

optional implementations which has a __dict__ that allows assignment
Or, should the implementations just live in the main backend?


Backend Implementations
[x] must be able to flatten model symbols to backend arrays,
[x] wrap backend arrays to model symbol, matching shape -- 
[ ] wrap and flatten must handle model numerics (float/numpy array) and backend numerics (if
different, eg casadi DM) and backend symbols
[ ] ideally, handle special case symmetric and dynamic flags for FreeSymbol and
[ ] MatchedSymbol if matched to symmetric/diagonal FreeSymbol
setting the values for outputs and intermediates

Couples to Model types -- knows all fields

who is responsible for:
filling in fields/._dataclass?
filling in model input/output attrs?


"""

"""
Figure out how to add DB storage -- maybe expect that to be a user choice (a decorator
or something)? Then user defined initializer could use it. Yeah, and ORM aspect just
takes advantage of model attributes just like backend implementation


I assume a user model/library code could inject an implementation to the backend?
not sure how to assign special numeric stuff, probably an submodel class on the model
based on NPSS discussion it's not really needed if it's done right 

For injecting a default implementation (e.g., new backend) this does work:

import casadi_implementations
casadi_implementations.ODESystem = 'a reference to check exists'

import condor as co

but probably should just figure out hooks to do that? Could create local dict of backend
that gets updated with backend.implementations at the top of this file, then libary/user
code could update it (add/overwrite)
"""


@dataclass
class BackendSymbolData:
    shape: tuple  # TODO: tuple of ints
    symmetric: bool
    diagonal: bool
    size: int = field(init=False)
    # TODO: mark size as computed? or replace with @property? can that be trivially cached?
    # currently computed by actual backend...

    def __post_init__(self, *args, **kwargs):
        n = self.shape[0]
        size = np.prod(self.shape)
        if self.symmetric:
            size = int(n * (n + 1) / 2)
        if self.diagonal:
            size = n

        if self.diagonal:
            assert size == self.shape[0]
        elif self.symmetric:
            assert len(shape) == 2
            assert self.shape[0] == self.shape[1]
        self.size = size

