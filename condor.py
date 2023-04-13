import numpy as np


# TODO: figure out how to make this an option/setting like django?
import casadi_backend as backend

class Expression:
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

    def _set_resolve_name(self):
        self._resolve_name = ".".join([self._model_name, self._name])

    def __init__(self, name=None, model=None):
        print('BaseDescriptor.__init__', self)

        self._name = name
        self._model = model
        if model:
            self._model_name = model.__class__.__name__
        else:
            self._model_name = None
        if model and name:
            self._set_resolve_name()
        self._count = 0 # shortcut for flattned size?
        self._container = []
        # actual storage -- best type? want name, reference to symbolic, possibly
        # additional metadata... maybe list of dict? at model instance creation, can
        # iterate or key values, 
        self._init_kwargs = {} 
        # subclasses must provide _init_kwargs for binding to sub-classes

    def bind(self, model_name, expression_type_name, **kwargs):
        # copy and re-assign name
        if not kwargs:
            kwargs = self._init_kwargs
        new = self.__class__(**kwargs)
        new._name = expression_type_name
        new._model_name = model_name
        new._set_resolve_name()
        return new


class Symbol(Expression):
    def __init__(self, name='', model=None, func=backend.symbol_generator, **fixed_kwargs):
        print('_condor_symbol_generator.__init__', self)
        super().__init__(name, model)
        self.func = func
        self.fixed_kwargs = fixed_kwargs
        print(fixed_kwargs)
        self._init_kwargs.update(dict(func=func,))
        if fixed_kwargs:
            self._init_kwargs.update(dict(fixed_kwargs=fixed_kwargs,))

    def __call__(self, n=1, m=1, symmetric=False, diagonal=False,  **kwargs):
        print("calling", self, "with args:", n, m, symmetric, diagonal)

        if diagonal:
            assert m == 1
        elif symmetric:
            assert n == m
        pass_kwargs = dict(
            name="%s_%d" % (self._resolve_name, self._count),
            n=n,
            m=m,
            symmetric=symmetric,
            diagonal=diagonal,
            **kwargs
        )
        if self.fixed_kwargs:
            pass_kwargs.update(**self.fixed_kwargs)
        print(pass_kwargs)
        if symmetric:
            size = int(n*(n+1)/2)
        else:
            size = n*m
        self._count += size
        out = self.func(**pass_kwargs)
        self._container.append(dict(
            name=None,
            symbol=out,
            n=n,
            m=m,
            symmetric=symmetric,
            diagonal=diagonal,
            size=size,
            **kwargs,
        ))
        return out


class FreeComputation(Expression):
    def __new__(cls, *args, **kwargs):
        print("init subclass free comp", cls, type(cls), args, kwargs)
        cls.known_attrs = cls.__dict__.copy()
        cls = super().__new__(cls, *args, **kwargs)
        return cls

    def __setattr__(self, name, value):
        if name.startswith('_'):
            print('skipping private _')
            super().__setattr__(name, value)
        elif name in self.known_attrs:
            print('skipping known')
            super().__setattr__(name, value)
        else:
            print("setting special for", self, ".", name, "=", value)
            print(f"owner={self._model_name}")


class MatchedComputation(Expression):
    def __init__(self, name='', model=None, matched_to=None):
        print('init', self)
        super().__init__(name, model)
        self._matched_to = matched_to
        self._init_kwargs = dict(matched_to=matched_to)
        print("matched to", matched_to)

    def __setitem__(self, key, value):
        print("setting item for", self, ":", key, "=", value)
        print(f"owner={self._model_name}")
        pass


# may not be  really necessary, place holder in case it is
class CondorClassDict(dict):
    def __getitem__(self, *args, **kwargs):
        #breakpoint()
        return super().__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        #breakpoint()
        return super().__setitem__(*args, **kwargs)


class ModelType(type):
    """
    Metaclass for Condor  model
    """

    @classmethod
    def __prepare__(cls, model_name, bases, **kwds):
        print("CondorModelType.__prepare__  for", model_name)
        sup_dict = super().__prepare__(cls, model_name, bases, **kwds)
        cls_dict = CondorClassDict(**sup_dict)

        dicts = [base.__dict__ for base in bases]

        for _dict in dicts:
            for k, v in _dict.items():
                if isinstance(v, Expression):
                    v_class = v.__class__
                    v_init_kwargs = v._init_kwargs.copy()
                    for init_k, init_v  in v._init_kwargs.items():
                        if isinstance(init_v, Expression):
                            v_init_kwargs[init_k] = cls_dict[init_v._name]
                    cls_dict[k] = v.bind(
                        model_name, expression_type_name=k, **v_init_kwargs
                    )
        print("end prepare for", model_name, cls_dict)
        return cls_dict

    def __call__(cls, *args, **kwargs):
        print("CondorModelType.__call__ for", cls)
        return super().__call__(cls, *args, **kwargs)

    def __new__(cls, name, bases, attrs, **kwargs):
        print("CondorModelType.__new__ for", name)
        print(attrs)
        new_cls = super().__new__(cls, name, bases, attrs, **kwargs)


        return new_cls

class Model(metaclass=ModelType):
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


