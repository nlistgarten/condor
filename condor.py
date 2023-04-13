import numpy as np


# TODO: figure out how to make this an option/setting like django?
import casadi_backend as backend

class CondorDescriptorType(object):
    pass


class BaseDescriptor(CondorDescriptorType):
    # these don't need to be descriptors, probably just a base class with
    # _set_resolve_name, called by CondorModelType.__new__
    # rename this class to CondorVariable? Expression? Something to capture both I and

    # maybe just separate out the matched and unmatched outputs into different
    # sub-classes?

    # assume all outputs are inherited?, base classes going backwards but within each
    # base preserve definition order. This allows placement of "output" in CondorModel
    # which everyone inherits, and it comes out last. Then algebraic system's
    # implicit_outputs, but I like that being implicit and explicit

    # the instances of variables added to the models may need to be replaced with
    # descriptor wrapper to allow dot setting of instance variables? if we even do
    # that... maybe not necessary either but could be an option

    # could use copy infrastructure with the same type of checking for references as
    # done in this version

    # general idea for handling implementation: during __new__, look up parents in
    # reverse order. can check backend.implementations (dict/module of model classname
    # to whatever class?)
    # I assume a user model could inject an implementation to the backend?
    # not sure how to assign special numeric stuff, probably an inner class on the model
    # based on NPSS discussion it's not really needed if it's done right 

    # once imp is found gets attached to class. then __init__ calls with those values
    # and so is actually an instance of class that has the accessed outputs, very much
    # like django. neat.

    # implementations can get everything they need based on the model field definitions
    # (ie. state, parameter, output, dot etc.) which is the point of them
    # and we could very easily just cache everything to a DB lol

    # can I do all the name clash protection that's needed? from child to parent class I
    # think definitely, and probably can't protect user model from over-writing self


    def __set_name__(self, owner, name):
        print("setting", name, "for", owner, "as", self)
        self._name = name
        self._owner = owner
        self._owner_name = self._owner.__class__.__name__
        self._set_resolve_name()

    def _set_resolve_name(self):
        self._resolve_name = ".".join([self._owner_name, self._name])

    def __get__(self, obj, objtype=None):
        print("getting", self._resolve_name, "as", self,  "for", obj)
        pass
        return 

    def __set__(self, obj, value):
        raise AttributeError(self._resolve_name + " cannot be set as " + self.__class__)

    def __call__(cls, *args, **kwargs):
        print("BaseDescriptor.__call__", cls, args, kwargs)

    def __init__(self, name='', owner=None):
        print('BaseDescriptor.__init__', self)
        if name and owner:
            self.__set_name__(owner, name)
        self._count = 0 # shortcut for flattned size?
        self._container = [] # actual storage
        self._init_kwargs = {}

    def _inherit(self, new_owner_name, **kwargs):
        # TODO: actually this isn't "inheriting" it's binding?
        if not kwargs:
            kwargs = self._init_kwargs
        new = self.__class__(**kwargs)
        new._name = self._name
        new._owner_name = new_owner_name
        new._set_resolve_name()
        return new


    #def __new__(cls, *args, **kwargs):
    #    print("BaseDescriptor.__new__", cls, args, kwargs)
    #    return cls()


class _condor_symbol_generator(BaseDescriptor):
    def __init__(self, func=backend.symbol_generator, name='', owner=None, **fixed_kwargs):
        print('_condor_symbol_generator.__init__', self)
        super().__init__(name, owner)
        self.func = func
        self.fixed_kwargs = fixed_kwargs
        self.init_kwargs = dict(func=func, fixed_kwargs=fixed_kwargs)

    def __call__(self, n=1, m=1, symmetric=False, diagonal=False,  **kwargs):
        print("calling", self, "with args:", n, m, symmetric, diagonal)

        pass_kwargs = dict(
            name="%s_%d" % (self._resolve_name, self._count),
            n=n,
            m=m,
            symmetric=symmetric,
            diagonal=diagonal,
            **self.fixed_kwargs,
            **kwargs
        )
        self._count += 1
        out = self.func(**pass_kwargs)
        #out = _condor_symbol_wrapped(out)
        self._container.append(out)
        # TODO: 
        # actually, this needs to be a descriptor container for the symbol
        # this allows dot setting of DynamicsModel parameter by a TrajectoryModel
        # or does the fact that the symbol generator is keeping a reference mean we can
        # access it? yes, requires __getattr__ on Model, could be fine



        return out


class _condor_computation(BaseDescriptor):
    def __init__(self, matched_to=None, name='', owner=None, ):
        print('init', self)
        super().__init__(name, owner)
        self._matched_to = matched_to
        self._init_kwargs = dict(matched_to=matched_to)
        print("matched to", matched_to)
        pass

    def __setitem__(self, key, value):
        print("setting item for", self, ":", key, "=", value)
        print(f"owner={self._owner_name}")
        pass

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            print("setting item for", self, ".", name, "=", value)
            print(f"owner={self._owner_name}")
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

class CondorComputationContainer(type):
    """
    An output container that allows named dot access and integer indexing
    """


class CondorModelType(type):
    """
    Metaclass for Condor  model
    """

    @classmethod
    def __prepare__(cls, name, bases, **kwds):
        print("CondorModelType.__prepare__  for", name)
        sup_dict = super().__prepare__(cls, name, bases, **kwds)
        cls_dict = CondorClassDict(**sup_dict)

        dicts = [base.__dict__ for base in bases]

        for _dict in dicts:
            for k, v in _dict.items():
                if (
                    isinstance(v, BaseDescriptor) or
                    isinstance(v, BaseDescriptor.__class__)
                ):
                    v_class = v.__class__
                    v_init_kwargs = v._init_kwargs.copy()
                    for init_k, init_v  in v._init_kwargs.items():
                        if isinstance(init_v, BaseDescriptor):
                            v_init_kwargs[init_k] = cls_dict[init_v._name]
                    cls_dict[k] = v._inherit(name, **v_init_kwargs)

        # TODO: don't love the level of coupling between CondorModel and descriptor
        # inheritance, but perhaps that's unavoidable. don't love the use of
        # _set_resolve_name, is it safe to assume name here will always match the
        # __set_name__ thae new descriptor sees? It should be, but slightly worried
        # assumption will break.

        print("end prepare for", name, cls_dict)
        return cls_dict

    def __call__(cls, *args, **kwargs):
        print("CondorModelType.__call__ for", cls)
        return super().__call__(cls, *args, **kwargs)

    def __new__(cls, name, bases, attrs, **kwargs):
        print("CondorModelType.__new__ for", name)
        print(attrs)
        new_cls = super().__new__(cls, name, bases, attrs, **kwargs)
        return new_cls

class CondorModel(metaclass=CondorModelType):
    # should output always be here?
    pass


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


