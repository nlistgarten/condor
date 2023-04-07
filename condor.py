import numpy as np


# TODO: figure out how to make this an option/setting like django?
import casadi_backend as backend
BACKEND_SYMBOL = sym.Symbol


class _base_condor_descriptor:
    def __set_name__(self, owner, name):
        print("setting", name, "for", owner, "as", self)
        self._name = name
        self._owner = owner
        self._resolve_name = ".".join([self._owner.__class__.__name__, self._name])

    def __get__(self, obj, objtype=None):
        print("getting", self._resolve_name, "as", self,  "for", obj)
        pass
        return 

    def __set__(self, obj, value):
        raise AttributeError(self._resolve_name + " cannot be set as " + self.__class__)




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
            name="%s%d_" % (self._resolve_name, self.count),
            n=n,
            **self.fixed_kwargs,
            **kwargs
        )
        return self.func(**pass_kwargs)


class _condor_computation(_base_condor_descriptor):
    def __init__(self, matched_to=None):
        print('init', self)
        self._matched_to = matched_to
        pass

    def __setitem__(self, key, value):
        print("setting item for", self, ":", key, "=", value)
        pass

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            print("setting item for", self, ".", name, "=", value)
        pass


# may not be  really necessary, place holder in case it is
class CondorClassDict(dict):
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
        cls_dict = CondorClassDict()

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

"""
do parent systems always write subsystems by getting all their output? I think so
Groups definitely need some type of naming scheme because all outputs get lifted up as
outputs (not disciplined/strict system encapsolation). Although my implementation of a
blockdiagram also does this by default?

get_val is nice for masking only what's needed, I guess we could use masks
what about multiple calls where some values don't change? need to reference that

in general symbolic and numeric references,

yeah, might need something
    fuel_flow_rate, thrust = propulsion(flight_condition, throttle)[
        propulsion.output.fuel_flow_rate,
        propulsion.output.thrust
    ]

which is ~ how the output df indexing should work? calls should genreally return the df,
indexable by number, symbolic object, etc.


"""

class CondorOutputContainer(type):
    """
    An output container that allows named dot access and integer indexing
    """

