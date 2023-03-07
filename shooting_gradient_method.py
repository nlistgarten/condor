import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

BACKEND_SYMBOL = sp.Symbol
class _null_name:
    pass


class _base_condor_descriptor:
    def __set_name__(self, owner, name):
        print("setting", name, "for", owner, "as", self)
        self.name = name
        self.owner = owner
        self.resolve_name = ".".join([self.owner.__class__.__name__, self.name])

    def __get__(self, obj, objtype=None):
        print("getting", self.resolve_name, "as", self,  "for", obj)
        pass
        return 

    def __set__(self, obj, value):
        raise AttributeError(self.resolve_name + " cannot be set as " + self.__class__)

def construct_explicit_matrix(name, n, m=1, symmetric=False, diagonal=0,
                              dynamic=False, **kwass):
    """
    construct a matrix of symbolic elements
    Parameters
    ----------
    name : string
        Base name for variables; each variable is name_ij, which
        admitedly only works clearly for n,m < 10
    n : int
        Number of rows
    m : int
        Number of columns
    symmetric : bool, optional
        Use to enforce a symmetric matrix (repeat symbols above/below diagonal)
    diagonal : bool, optional
        Zeros out off diagonals. Takes precedence over symmetry.
    dynamic : bool, optional
        Whether to use sympy.physics.mechanics dynamicsymbol. If False, use
        sp.symbols
    kwargs : dict
        remaining kwargs passed to symbol function
    Returns
    -------
    matrix : sympy Matrix
        The Matrix containing explicit symbolic elements
    """
    if dynamic:
        symbol_func = vector.dynamicsymbols
    else:
        symbol_func = sp.symbols

    if n != m and (diagonal or symmetric):
        raise ValueError("Cannot make symmetric or diagonal if n != m")

    if diagonal:
        return sp.diag(
            *[symbol_func(
                name+'_{}{}'.format(i+1, i+1), **kwass) for i in range(m)])
    else:
        matrix = sp.Matrix([
            [symbol_func(name+'_{}_{}'.format(j+1, i+1), **kwass)
             for i in range(m)] for j in range(n)
        ])

        if symmetric:
            for i in range(1, m):
                for j in range(i):
                    matrix[j, i] = matrix[i, j]

        return matrix


class _condor_symbol_generator(_base_condor_descriptor):
    def __init__(self, func=construct_explicit_matrix, **fixed_kwargs):
        print('init', self)
        self.func = func
        self.fixed_kwargs = fixed_kwargs
        self.count = 0
        self.children = []

    def __call__(self, n, **kwargs):
        print("calling for", self)
        pass_kwargs = dict(
            name="%s%d_" % (self.resolve_name, self.count),
            n=n,
            **self.fixed_kwargs,
            **kwargs
        )
        return self.func(**pass_kwargs)



class _condor_computation(_base_condor_descriptor):
    def __init__(self, matched_to=None):
        print('init', self)
        pass

    def __setitem__(self, key, value):
        print("setting item for", self, ":", key, "=", value)
        pass


# not really necessary, place holder in case it is
class CondorDict(dict):
    def __getitem__(self, *args, **kwargs):
        #breakpoint()
        return super().__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        #breakpoint()
        return super().__setitem__(*args, **kwargs)

from enum import _is_dunder

class CondorModelType(type):
    """
    Metaclass for Condor  model
    """

    @classmethod
    def __prepare__(cls, name, bases, **kwds):
        print("preparing for", name)
        sup_dict = super().__prepare__(cls, name, bases, **kwds)
        cls_dict = CondorDict()

        dicts = [sup_dict] + [base.__dict__ for base in bases]
        for _dict in dicts:
            cls_dict.update({
                k: v for k, v in  _dict.items()
                if (isinstance(v, _base_condor_descriptor) or (_dict is sup_dict))
            })
        return cls_dict

    def __new__(cls, name, bases, attrs, **kwargs):
        print("__new__ for", name)
        new_cls = super().__new__(cls, name, bases, attrs, **kwargs)
        return new_cls

class DynamicsModel(metaclass=CondorModelType):
    print("starting user code for CondorModel class")
    state = _condor_symbol_generator()
    parameter = _condor_symbol_generator()
    state_rate = _condor_computation(state)
    output = _condor_computation()
    print("ending user code for TrajectoryModel class")


class LTI(DynamicsModel):
    print("starting user code for LTI class")
    n = 2
    m = 1
    z = n + m
    # p, v = state()
    x = state(n)
    C = state(n=n,m=n, symmetric=True)
    A = np.array([
        [0, 1],
        [0, 0],
    ])

    # A = parameter(n,n)
    # B = parameter(n,m)
    # K = parameter(m,n)

    print(A @ x)
    state_rate[x] = A@x #(A - B @ K) @ x
    state_rate[C] = A@C + C@A.T
    print("ending user code for LTI class")


