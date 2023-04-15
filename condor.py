import numpy as np
from dataclasses import dataclass


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

    def __init__(self, name='', model=None, symbol_container=None):
        self._name = name
        self._model = model
        if model:
            self._model_name = model.__class__.__name__
        else:
            self._model_name = ''
        self._set_resolve_name()
        self._count = 0 # shortcut for flattned size?
        self._containers = []
        self._symbol_container = symbol_container
        self._init_kwargs = dict()
        # subclasses must provide _init_kwargs for binding to sub-classes
        # TODO: can this just be taken from __init__ kwargs easily?
        # or __init_subclass__ hook?


    def bind(self, name, model,):
        # bind an existing expression class to a model
        self._name = name
        self._model = model
        if self._model_name:
            if self._model_name != model.__name__:
                raise ValueError("attempting to bind to a class that wasn't inherited")
        else:
            self._model_name = model.__name__
        self._set_resolve_name()

    def inherit(self, model_name, expression_type_name, **kwargs):
        # copy and re-assign name -- used in ModelType.__prepare__ before model is
        # available
        if not kwargs:
            kwargs = self._init_kwargs
        new = self.__class__(**kwargs)
        new._name = expression_type_name
        new._model_name = model_name
        new._set_resolve_name()
        return new

    # TODO: replace with a descriptor that can be dot accessed?
    def list_of(self, field_name):
        return [getattr(container, field_name) for container in self._containers]

    def __repr__(self):
        if self._resolve_name:
            return self._resolve_name

    def get(self, **kwargs):
        """
        return list of expression containers where every field matches kwargs
        """
        # TODO: what's the lightest-weight way to be able query? these should get called
        # very few times, so hopefully don't need to stress too much about
        # implementation
        # would be nice to be able to follow references, etc. can match be used?
        items = []
        for item in self._containers:
            this_item = True
            for field_name, field_value in kwargs.items():
                this_item = this_item & (getattr(item, field_name) is field_value)
                if not this_item:
                    break
            if this_item:
                items.append(item)
        if len(items) == 1:
            return items[0]
        return items

    def create_container(self, **kwargs):
        kwargs.update(dict(expression_type=self))
        self._containers.append(self._symbol_container(**kwargs))

@dataclass
class BaseContainer:
    expression_type: Expression

@dataclass
class SymbolContainer(BaseContainer):
    # TODO: this data structure is  repeated in 2 other places, Symbol.__call__ and
    # symbol_generator. It should be dry-able, but validation is complicated..
    name: str
    symbol: backend.symbol_class
    n: int
    m: int
    symmetric: bool
    diagonal: bool
    size: int

class Symbol(Expression):
    # TODO: is it possible to inherit the kwargs name, model, symbol_container?
    # repeated... I guess kwargs only? but like name as optional positional. maybe
    # additional only kwargs? how about update default? Should the default be done by
    # name? e.g., ExpressionSubclass.__name__ + 'Container'? See FreeComputation below
    def __init__(
        self, name='', model=None,
        func=backend.symbol_generator,
    ):
        # TODO: should model types define the extra fields here?
        # I guess these could be metaclasses, and things like DynamicsModel are really
        # model types? could define state by creating inner class definition that
        # subclasses this to define extra fields (e.g., optimization variable or
        # implicit outputs getting ref, scaling, etc)
        # this defaults with n, m, symmetric, etc.
        # but actually do need to return just symbol from user code. so maybe libraries
        # define symbolcontainer subclasses, __call__ takes the fields as args?
        # they can just be dataclasses.
        # having DynamicModelType be a metaclass (or, allowing/encouraging library
        # Model(Types) to write a metaclass might allow useful hooks?j
        super().__init__(name, model, symbol_container=SymbolContainer)
        self.func = func
        self._init_kwargs.update(dict(func=func))

    def __call__(self, n=1, m=1, symmetric=False, diagonal=False,):
        print("calling", self, "with args:", n, m, symmetric, diagonal)

        if diagonal:
            assert m == 1
        elif symmetric:
            assert n == m
        pass_kwargs = dict(
            name="%s_%d" % (self._resolve_name, len(self._containers)),
            n=n,
            m=m,
            symmetric=symmetric,
            diagonal=diagonal,
            #**kwargs
        )
        #if self.fixed_kwargs:
        #    pass_kwargs.update(**self.fixed_kwargs)
        #print(pass_kwargs)
        if symmetric:
            size = int(n*(n+1)/2)
        else:
            size = n*m
        self._count += size
        out = self.func(**pass_kwargs)
        self.create_container(
            name=None,
            symbol=out,
            n=n,
            m=m,
            symmetric=symmetric,
            diagonal=diagonal,
            size=size,
        )
        return out

@dataclass
class FreeComputationContainer(BaseContainer):
    name: str
    symbol: backend.symbol_class

class FreeComputation(Expression):
    def __init__(self, name='', model=None,):
        super().__init__(name=name, model=model, symbol_container=FreeComputationContainer)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            # I would like it if I could DRY-ly check for "known" class attributes to
            # skip/raise clash, but can't tell the difference between user code and
            # library code, latter does need to set them, also not clear when/where to
            # collect. So all core variable names on Expression must be prefixed by _
            print('skipping private _')
            super().__setattr__(name, value)
        else:
            print("setting special for", self, ".", name, "=", value)
            print(f"owner={self._model_name}")
            self.create_container(
                name=name,
                symbol=value,
            )

@dataclass
class MatchedComputationContainer(BaseContainer):
    match: backend.symbol_class
    symbol: backend.symbol_class

class MatchedComputation(Expression):
    def __init__(self, matched_to=None, model=None, name=''):
        super().__init__(name, model, symbol_container=MatchedComputationContainer)
        self._matched_to = matched_to
        self._init_kwargs.update(dict(matched_to=matched_to))
        print("matched to", matched_to)

    def __setitem__(self, key, value):
        print("setting item for", self, ":", key, "=", value)
        print(f"owner={self._model_name}")
        if isinstance(key, backend.symbol_class):
            match = self._matched_to.get(symbol=key)
            if isinstance(match, list):
                raise ValueError
            self.create_container(
                match=match,
                symbol=value,
            )


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
                    cls_dict[k] = v.inherit(
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
        # what gets manipulated on attrs before new_cls?
        new_cls = super().__new__(cls, name, bases, attrs, **kwargs)

        # find implementation from backend/bases, create and assign (called by Model.__init__)
        for base in bases:
            break

        symbol_instances = []
        for attr_name, attr_val in attrs.items():
            # process name clahees on attr_name? can't protect against internal
            # over-write (maybe user intends to do that) but can raise error for
            # over-writing parent, e.g., creating a "state" variable in a DynamicModel
            # subclass
            # TODO: replace ifs with match?
            if isinstance(attr_val, Expression):
                # need hook to define which Expression types form args of imp.call and in
                # what order? Or __init__ just take kwargs and it's up to the
                # implementation/backend to validate?
                attr_val.bind(attr_name, new_cls)
            if isinstance(attr_val, Symbol):
                # TODO: assumes Symbol is the base of all inputs to the model functions?
                # is there a better way to declare that?
                # create fields on self?
                symbol_instances.append(attr_val)
            if isinstance(attr_val, backend.symbol_class):
                for symbol_instance in symbol_instances:
                    container = symbol_instance.get(symbol=attr_val)
                    # not a list (empty or len > 1)
                    if isinstance(container, BaseContainer):
                        container.name = attr_name
                # add intermediate computations?
            # TODO: add hook so e.g., DynamicModel (or consumer) can deal with inner
            # class for events. Not sure if that's the same as a deferred
            # subsystem -- 

            # TODO: process/inherit options for backend -- similar to django Meta
            # processing





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


class DynamicsModel(Model):
    """
    DynamicsModel

    need input? (e.g., time varying paremeter) -> a parent class pushes input in
    or skip, assume plant always "pulls" input in

    in this context is "output" always point-wise in time? I think so, and it's a
    trajectory analysis model's outputs that define overall output

    fundamentally need to decide how different computations play. Maybe "output" is the
    base on CondorModel, and it is always the one that is returned by calling -- would
    be consistent way to pull controllers, measurements, etc from the plant model
    need API to specify which of the symbols are used as arguments to call?hkk

    then trajectory analysis model's IMP can 

    """
    print("starting app code for DynamicsModel class")

    # TODO: this needs its own descriptor type? OR do we want user classes to do
    # t = DynamicsModel.independent_variable ? That would allow 
    independent_variable = backend.symbol_generator('t')
    state = Symbol()
    parameter = Symbol()
    dot = MatchedComputation(state)
    output = FreeComputation()
    print("ending app code for DynamicsModel class")


