import numpy as np
from dataclasses import dataclass, make_dataclass
from enum import Enum


# TODO: figure out how to make this an option/setting like django?
import casadi_backend as backend
"""
Backend:
[x] provide symbol_generator for creating backend symbol repr
[x] symbol_class for isinstance(model_attr, backend.symbol_class
[x] name which is how backend options on model is identified
Do we allow more complicated datastructures? like models, etc.

optional implementations which has a __dict__ that allows assignment


Backend Implementations
[x] must be able to flatten model symbols to backend arrays,
[ ] wrap backend arrays to model symbol, matching shape -- 
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
not sure how to assign special numeric stuff, probably an inner class on the model
based on NPSS discussion it's not really needed if it's done right 

For injecting a default implementation (e.g., new backend) this does work:

import casadi_implementations
casadi_implementations.ODESystem = 'a reference to check exists'

import condor as co

but probably should just figure out hooks to do that? Could create local dict of backend
that gets updated with backend.implementations at the top of this file, then libary/user
code could update it (add/overwrite)

"""

class Direction(Enum):
    """
    Used to indicate the direction of a Symbol relative to a model
    MatchedField may need to become MatchedSymbol and also use direction -- will
    be useful for DAE models, etc.
    """
    output = -1
    internal = 0
    input = 1

class FieldValues:
    pass

class Field:
    """
    """

    def _set_resolve_name(self):
        self._resolve_name = ".".join([self._model_name, self._name])

    def __init_subclass__(cls, symbol_class=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.symbol_class = symbol_class

    def __init__(
        self, name='', model=None, direction=Direction.internal, inherit_from=None
    ):
        # TODO: currently, AssignedField types are defined using setattr, which needs
        # to know what already exists. The attributes here, like name, model, count,
        # etc, don't exist until instantiated. Currently pre-fix with `_` to mark as
        # not-a-computation, but I guess I could just use symbol_class on value instead
        # of checking name? Anyway, really don't like needing to prefix all Field
        # attributes with _ because of AssignedField... 


        self._name = name
        self._model = model
        self._direction = direction
        if model:
            self._model_name = model.__class__.__name__
        else:
            self._model_name = ''
        self._set_resolve_name()
        self._count = 0 # shortcut for flattned size?
        self._symbols = []
        self._init_kwargs = dict(direction=direction)
        self._inherits_from = inherit_from
        # subclasses must provide _init_kwargs for binding to sub-classes
        # TODO: can this just be taken from __init__ kwargs easily?  or
        # __init_subclass__ hook? definitely neds to be DRY'd up 


    def bind(self, name, model,):
        """
        bind an existing Field instance to a model
        """
        self._name = name
        self._model = model
        if self._model_name:
            if self._model_name != model.__name__:
                raise ValueError("attempting to bind to a class that wasn't inherited")
        else:
            self._model_name = model.__name__
        self._set_resolve_name()

    def inherit(self, model_name, field_type_name, **kwargs):
        """
        copy and re-assign name -- used in ModelType.__prepare__ before model is
        available
        """

        # TODO: could use copy infrastructure with the same reference checking Model is
        # currently doing
        if not kwargs:
            kwargs = self._init_kwargs
        new = self.__class__(**kwargs, inherit_from=self)
        new._name = field_type_name
        new._model_name = model_name
        new._set_resolve_name()
        return new

    # TODO: replace with a descriptor that can be dot accessed?
    def list_of(self, field_name):
        return [getattr(symbol, field_name) for symbol in self._symbols]

    def __repr__(self):
        if self._resolve_name:
            return f"<{self.__class__.__name__}: {self._resolve_name}>"

    def get(self, **kwargs):
        """
        return list of field symbols where every field matches kwargs
        """
        # TODO: what's the lightest-weight way to be able query? these should get called
        # very few times, so hopefully don't need to stress too much about
        # implementation
        # would be nice to be able to follow references, etc. can match be used?
        items = []
        for item in self._symbols:
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

    def create_symbol(self, **kwargs):
        kwargs.update(dict(field_type=self))
        self._symbols.append(self.symbol_class(**kwargs))

    def create_dataclass(self):
        # TODO: do processing to handle different field types
        fields = [(symbol.name, float) for symbol in self._symbols]
        name = make_class_name([self._model_name, self._name])
        self._dataclass = make_dataclass(
            name,
            fields,
            bases=(FieldValues,),
        )

    def __iter__(self):
        for symbol in self._symbols:
            yield symbol

def make_class_name(components):
    separate_words = ' '.join([comp.replace('_', ' ') for comp in components])
    # use pascal case from https://stackoverflow.com/a/8347192
    return ''.join(word for word in separate_words.title() if not word.isspace())

@dataclass
class BaseSymbol:
    field_type: Field
    # TODO: Should it be renamed to something related to "backend" since it's really the
    # representation of backend.symbol_class? Should we support multiple back-end 
    # representations simultaneously?
    backend_repr: backend.symbol_class
    # TODO: should backend extract field data like shape, etc that all symbols
    # have? matches ones probably take the the metadata of their match, no need to
    # repeat. For general assigned field, probably lose them anyway -- can probably
    # figure out size, but not if diagonal/symetric is preserved; are those ever
    # expected to preserved except when matched?

    def __repr__(self):
        return f"<{self.field_type._resolve_name}: {self.name}>"

# TODO: IndependentInputSymbol and Depdendent? Or is this okay?

class IndependentSymbol(BaseSymbol):
    pass

@dataclass(repr=False)
class FreeSymbol(IndependentSymbol):
    # TODO: this data structure is  repeated in 2 other places, Symbol.__call__ and
    # symbol_generator. It should be dry-able, but validation is complicated..
    # one idea: where symbols gets instantiated, use **kwargs to try constructing,
    # Symbol class will at least validate field name assignment. Then Field
    # subclass does model-level validation.
    # or can the symbol itself perform validation? It has field_type back
    # referene, so it could do it...
    name: str
    n: int
    m: int
    # TODO: convert to shape? Then Symbol __call__ does one set of validation to ensure
    # consistency of model, backend can do addiitonal validation (e.g., casadi doesn't
    # support more than 2D due to matlab). Then again, symmetric and diagonal become
    # less clear, maybe Symbols are also no more than 2D.
    # but will eventually want N-D table lookup, knots and coefficients as parameters
    # for table lookups that fed from a model, use partial to fix them for models that
    # fed ? Use None for any dimension to follow broadcasting rules? Could even allow
    # ellipse
    # maybe have tables take knots and coeffs as an assigned input? Could similarly have
    # LTI system generator take matrices as assigned inputs? Special AssignmentField
    # that takes data instead of symbol. For fixed tables, not models that get optimized
    # Actually, table knots match to inputs and coefficients match to outputs.
    # so only LTI needs an assigned subclass for data/matrices that has fixed
    # attributes?

    # Then if bounds are here, must follow broadcasting rules

    symmetric: bool
    diagonal: bool
    size: int
    # TODO: mark size as computed? or replace with @property? can that be trivially cached?


class IndependentField(Field):
    """
    Fields that are arguments to implementation function:
    Better: used to define fields? Free?
    Symbol, deferred subsystems
    others?
    boundedsymbol adds upper and lower attribute
    or just always have bounds and validate them for Functions but pass on to solvers,
    etc? default bounds can be +/- inf

    No, really it's fields that are generated e.g., x = state()
    vs things that are assigned eg output.z = ...
    Then again, getattr and getitem can also be used to "generate" a symbol by name


    DependentFields instead of ComputedFields
    These have things (Actual Fields) assigned to them 
    free, matched
    Maybe something that just creates data structure? Could be useful for inheriting
    Aviary cleanly. Might need matching subclass of Indep.


    I also kind of think the three we have are the only three that will be developed,
    especially if "matched" can be any generalized to patterns? oh but also deferred
    symbol, eventually events, modes. But are those really special types of
    sub-models?

    And (especially with advanced setattr) Free can be used create a dotable data
    structure heirarchy?

    """
    pass


class FreeField(IndependentField, symbol_class=FreeSymbol):
    # TODO: is it possible to inherit the kwargs name, model, symbol_class?
    # repeated... I guess kwargs only? but like name as optional positional. maybe
    # additional only kwargs? how about update default? Should the default be done by
    # name? e.g., FieldSubclass.__name__ + 'Symbol'? See AssignedField below
    def __init__(
        self, direction=Direction.input, name='', model=None, inherit_from=None,
    ):
        # TODO: should model types define the extra fields here?
        # I guess these could be metaclasses, and things like DynamicsModel are really
        # model types? could define state by creating inner class definition that
        # subclasses this to define extra fields (e.g., optimization variable or
        # implicit outputs getting ref, scaling, etc)
        # this defaults with n, m, symmetric, etc.
        # but actually do need to return just symbol from user code. so maybe libraries
        # define symbol subclasses, __call__ takes the fields as args?
        # they can just be dataclasses.
        # having DynamicModelType be a metaclass (or, allowing/encouraging library
        # Model(Types) to write a metaclass might allow useful hooks?j
        super().__init__(
            name=name, model=model, direction=direction, inherit_from=inherit_from
        )

    def __call__(self, n=1, m=1, symmetric=False, diagonal=False,):
        if diagonal:
            assert m == 1
        elif symmetric:
            assert n == m
        pass_kwargs = dict(
            name="%s_%d" % (self._resolve_name, len(self._symbols)),
            n=n,
            m=m,
            symmetric=symmetric,
            diagonal=diagonal,
            #**kwargs
        )
        if symmetric:
            size = int(n*(n+1)/2)
        else:
            size = n*m
        self._count += size
        out = backend.symbol_generator(**pass_kwargs)
        self.create_symbol(
            name=None,
            backend_repr=out,
            n=n,
            m=m,
            symmetric=symmetric,
            diagonal=diagonal,
            size=size,
        )
        return out



@dataclass(repr=False)
class AssignedField(BaseSymbol):
    name: str

class AssignedField(Field, symbol_class=AssignedField):
    def __init__(
        self, direction=Direction.output, name='', model=None, inherit_from=None
    ):
        super().__init__(
            name=name, model=model, direction=direction, inherit_from=inherit_from
        )

    def __setattr__(self, name, value):
        if name.startswith('_'):
            print('skipping private _')
            super().__setattr__(name, value)
        else:
            print("setting special for", self, ".", name, "=", value)
            print(f"owner={self._model_name}")
            self.create_symbol(
                name=name,
                backend_repr=value,
                # TODO: I guess if this accepts model instances, it becomes recursive to
                # allow dot access to sub systems? Actually, this breaks the idea of
                # both system encapsolation and implementations. So don't do it, but
                # doument it. Can programatically add sub-system outputs though. For
                # these reasons, ditch intermediate stuff.
            )
            super().__setattr__(name, self._symbols[-1])

@dataclass(repr=False)
class MatchedField(BaseSymbol):
    match: BaseSymbol # match to the Symbol instance

    def update_name(self):
        self.name = '__'.join([self.field_type._name, self.match.name])

class MatchedField(Field, symbol_class=MatchedField):
    def __init__(
        self, matched_to=None, direction=Direction.output, name='', model=None, inherit_from=None
    ):
        """
        matched_to is Field instance that this MatchedField is matched to.
        """
        super().__init__(
            name=name, model=model, direction=direction, inherit_from=inherit_from
        )
        self._matched_to = matched_to
        self._init_kwargs.update(dict(matched_to=matched_to))
        print("matched to", matched_to)

    def __setitem__(self, key, value):
        print("setting item for", self, ":", key, "=", value)
        print(f"owner={self._model_name}")
        if isinstance(key, backend.symbol_class):
            match = self._matched_to.get(backend_repr=key)
            if isinstance(match, list):
                raise ValueError
            self.create_symbol(
                match=match,
                backend_repr=value,
            )

    def __getitem__(self, key):
        """
        get the matched symbol by the matche's name, backend symbol, or symbol
        """
        if isinstance(key, backend.symbol_class):
            match = self._matched_to.get(backend_repr=key)
        elif isinstance(key, str):
            match = self._matched_to.get(name=key)
        elif isinstance(key, BaseSymbol):
            match = key
        else:
            raise ValueError

        item = self.get(match=match)
        if isinstance(item, list) and self._direction != Direction.input:
            raise ValueError
        return item



# may not be  really necessary, place holder in case it is
class CondorClassDict(dict):
    def __getitem__(self, *args, **kwargs):
        #breakpoint()
        return super().__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        #breakpoint()
        return super().__setitem__(*args, **kwargs)

class Options:
    """
    Class mix-in to flag back-end options. Define an inner class on the model that
    inherits from Options that over-writes options for the backend's implementation
    (which generates a callable). Model will transparently pass attributes of Options
    subclass. Possibly only convention, but name the subclass according to the backend.
    Multiple option sub-classes provide options depending on project's default backend. 
    Single option sub-class forces back end for this model?
    OR flag, if tihs backend then these options, otherwise no options (rely on defaults)
    
    implementation, if provided, should be a callable that takes a model and possible
    options and returns a callable that evaluates the model

    All backend implementations must ship with reasonable defaults.

    options should not start with __

    Example for inheritance to keep configuration DRY

    class UpcycleSolverOptions(Options):
        warm_start = True
        bound_behavior = backend.algebraicsystem.bound_behavior_options.L2
        exact_hessian = False


    class Upsolver(AlgebraicSystem):

        class Casadi(UpcycleSolverOptions):
            pass

    class MostUpsolvers(Upsolver):
        ... (define model)
        # automatically inherit UpcycleSolverOptions settings for Casadi backend

    class ParicularUpsolver(Upsolver):
        ... (define model)

        class Casadi(UpcycleSolverOptions):
            # over write specific settings, everything else should be inherited
            exact_hessian = True


    """
    pass


class ModelType(type):
    """
    Metaclass for Condor  model
    """

    # No dot setting, use func tool `partial` (or just define a dict that gets **passed
    # in). Dot accessing done by instance datastructure.
    # Do we want to provide magic for repeatedly passed-down variables? No. Keeps it
    # explicit, and there's enough convenience it shouldn't be a problem.

    # once imp is found gets attached to class. then __init__ calls with those values
    # and so is actually an instance of class that has the accessed outputs, very much
    # like django. neat.

    # can I do all the name clash protection that's needed? from child to parent class I
    # think definitely, and probably can't protect user model from over-writing self


    @classmethod
    def __prepare__(cls, model_name, bases, **kwds):
        print("CondorModelType.__prepare__  for", model_name)
        sup_dict = super().__prepare__(cls, model_name, bases, **kwds)
        cls_dict = CondorClassDict(**sup_dict)

        dicts = [base.__dict__ for base in bases]

        for _dict in dicts:
            for k, v in _dict.items():
                if isinstance(v, Field):
                    v_class = v.__class__
                    v_init_kwargs = v._init_kwargs.copy()
                    for init_k, init_v  in v._init_kwargs.items():
                        if isinstance(init_v, Field):
                            v_init_kwargs[init_k] = cls_dict[init_v._name]
                    cls_dict[k] = v.inherit(
                        model_name, field_type_name=k, **v_init_kwargs
                    )
        print("end prepare for", model_name, cls_dict)
        return cls_dict

    def __call__(cls, *args, **kwargs):
        print("CondorModelType.__call__ for", cls)
        return super().__call__(cls, *args, **kwargs)

    def __repr__(cls):
        if cls._parent_name:
            return f"<{cls._parent_name}: {cls.__name__}>"
        else:
            return f"<{cls.__name__}>"

    def __new__(cls, name, bases, attrs, **kwargs):
        # TODO: replace if tree with match cases?

        # TODO: add hooks so e.g., DynamicModel (or consumer) can deal with inner
        # class for events. Not sure if that's the same as a deferred subsystem -- 
        # can hooks be classmethods on the Model subclass? I suspect at least post_new
        # hook will be dot callable, pre_new hook can be taken from attrs?
        # need to return a status (defined by enum?) of continue rest of if-tree or
        # continue to next attr
        pre_super_new_hook = attrs.pop('pre_super_new_attr_hook', None)
        # pre_new_attr_hook(super_attrs, attr_name, attr_val)
        post_super_new_hook = attrs.pop('post_super_new_attr_hook', None)
        # pre_new_attr_hook(new_cls, attr_name, attr_val)

        print("CondorModelType.__new__ for", name)
        print(attrs)
        # what gets manipulated on attrs before new_cls?

        super_attrs = {}

        if bases:
            super_attrs['_parent_name'] = bases[0].__name__
        else:
            super_attrs['_parent_name'] = ''

        backend_options = {}
        free_fields = [] # used to find free symbols and replace with symbol
        matched_fields = []

        # will be used to pack arguments in (inputs) and unpack results (output)
        input_fields = []
        output_fields = []
        internal_fields = []
        input_names = []
        output_names = []

        super_attrs.update(dict(
            input_fields = input_fields,
            output_fields = output_fields,
            input_names = input_names,
            output_names = output_names,
        ))
        # TODO: create dataclass for each output type, assign as class attribute?
        # actually, does the implementation get this? It stuffs and returns on call
        # well, moel instance still needs to assign result to attr on self

        # can similar structure be used for inputs? Don't want to make it hard to
        # instantiate -- and most Symbol inputs have a convenient name that's free.
        # What about matched symbol inputs like dot for DAE? 


        for attr_name, attr_val in attrs.items():
            # TODO: process name clahees on attr_name? can't protect against internal
            # over-write (maybe user intends to do that) but can raise error for
            # over-writing parent, e.g., creating a "state" variable in a DynamicModel
            # subclass
            # need to have protected names? I guess whatever we're doing for Options and
            # convenience fields like input_fields, input_fields, etc?


            # TODO: hook for pre- super_new call

            pass_attr = True

            if isinstance(attr_val, type) and issubclass(attr_val, Options):
                backend_options[attr_name] =  attr_val
                continue
            if isinstance(attr_val, FreeField):
                free_fields.append(attr_val)
            if isinstance(attr_val, MatchedField):
                matched_fields.append(attr_val)
            if isinstance(attr_val, Field):
                # TODO: may need to deal with both cases that might occur: 
                # ModelType defining field OR user calling a model? Or should output
                # datastructure not be a subclass of BaseSymbol? Probably not!
                # and probably don't even need to handle them especially
                # can differentiate model definition by logic on bases -- can probably
                # add helper attributes to Field and/or flag from bases
                if attr_val._direction == Direction.input:
                    input_fields.append(attr_val)
                if attr_val._direction == Direction.output:
                    output_fields.append(attr_val)
                if attr_val._direction == Direction.internal:
                    internal_fields.append(attr_val)

            if isinstance(attr_val, backend.symbol_class):
                # from a FreeField
                known_symbol_type = False
                for free_field in free_fields:
                    symbol = free_field.get(backend_repr=attr_val)
                    # not a list (empty or len > 1)
                    if isinstance(symbol, BaseSymbol):
                        known_symbol_type = True
                        if symbol.name and symbol.name != attr_name:
                            raise NameError(f"Symbol on {free_field} has name {symbo.name} but assigned to {attr_name}")
                        symbol.name = attr_name
                        attr_val = symbol
                        pass_attr = False
                        break
                if not known_symbol_type:
                    print("unknown symbol type", attr_name, attr_val)
                    # Hit by DynamicModel.independent_variable
            #if isinstance(attr_val, Model):
            # TODO: from a sub-model (or deferred model?) call

            if pass_attr:
                check_attr_name(attr_name, attr_val, super_attrs, bases)
                super_attrs[attr_name] = attr_val

        for matched_field in matched_fields:
            for matched_symbol in matched_field:
                matched_symbol.update_name()

        for input_field in input_fields:
            for in_symbol in input_field:
                in_name = in_symbol.name
                check_attr_name(in_name, in_symbol, super_attrs, bases)
                super_attrs[in_name] = in_symbol
                input_names.append(in_name)
            input_field.create_dataclass()

        for internal_field in internal_fields:
            internal_field.create_dataclass()

        for output_field in output_fields:
            for out_symbol in output_field:
                out_name = out_symbol.name
                output_names.append(out_name)
                if not isinstance(out_symbol, IndependentSymbol):
                    # this symbol was not already attached to model
                    check_attr_name(out_name, out_symbol, super_attrs, bases)
                    super_attrs[out_name] = out_symbol
            output_field.create_dataclass()

        # TODO: validate backend_options and add to super_attrs
        # could have a reserved keyword "implementation" on options that defaults to
        # None, reference to an implementation to allow user provided ones? Maybe that's
        # the best way to do it, even for model types that condor ships with? if
        # inheritance works properly, can easily over-write per project, etc.
        new_cls = super().__new__(cls, name, bases, super_attrs, **kwargs)

        implementation = None

        if backend_options and backend.name in backend_options:
            backend_option = {
                k: v 
                for k, v in backend_options[backend.name].__dict__.items()
                if not k.startswith('__')
            }
            implementation = backend_option.pop('implementation', None)
            if implementation is not None:
                # inject so subclasses get this implementation
                # TODO: a better way? registration? etc?
                backend.implementation.__dict__[name] = implementation

            # TODO: other validation?
            # TODO: inherit options? No, defaults come from implementation itself
        else:
            backend_option = {}

        if implementation is None:
            for base in bases:
                if (
                    isinstance(base, ModelType) and 
                    (base.__name__ in backend.implementations.__dict__)
                ):
                    # TODO: or are these directly on backend along with symbol_Class,
                    # symbol_generator, and other things are in a utils submodule?
                    implementation = getattr(
                        backend.implementations,
                        base.__name__
                    )
                    break

        if implementation is not None:
            new_cls.implementation = implementation(new_cls, **backend_option)

        for attr_name, attr_val in attrs.items():
            # TODO: hook for post-- super_new call
            if isinstance(attr_val, Field):
                # need hook to define which Field types form args of imp.call and in
                # what order? Or __init__ just take kwargs and it's up to the
                # implementation/backend to validate?
                # this one definitely needs to be done after super new since we're
                # binding the class object
                attr_val.bind(attr_name, new_cls)

                # TODO: is it helpful to collect the fields? maybe not since
                # implemntations know what fields exist


            # TODO: add dependent variable data structure? they'll have to be constructed so implementation
            # data can be unpacked to represent model (chain dot accessors)
            # Solver type systems have solver variables that are "FreeField" like but
            # outputs?


        return new_cls


def check_attr_name(attr_name, attr_val, super_attrs, bases):
    # TODO: this needs to also be called to check on bases dict to prevent creating a
    # symbol with the name of state? I think?

    if attr_name in ['__module__', '__qualname__']:
        return

    if len(bases) == 1 and bases[0] == Model:
        # Skip model types
        # TODO: is there a better way to capture it's a model type? No symbols, but
        # fields?
        return

    # TODO: if user-defined field starts with _, raise value error?

    in_bases = False
    for base in bases:
        # TODO: only need to check the parent, not all bases?
        if (
            attr_name in base.__dict__ and
            not getattr(attr_val, '_inherits_from', None) == getattr(base, attr_name)
        ):
            in_bases = True
            break

    if (attr_name in super_attrs) or in_bases:
        clash_source = None
        if in_bases:
            clash_source = base
        elif hasattr(super_attrs[attr_name], 'inherits_from'):
            clash_source = super_attrs[attr_name].inherits_from
        if clash_source:
            clash_string = f" on {clash_source}"
        else:
            clash_string = ""
        raise NameError(f"Attempting to assign attribute {attr_name} which already exists{clash_string}")

class Model(metaclass=ModelType):

    def __init__(self, *args, **kwargs):
        cls = self.__class__
        # bind *args and **kwargs to to appropriate signature
        # pack into dot-able storage
        # call cls.imp, fill datastructures for both inputs and outputs
        # unpack into instance attrs

        # so all inputs and outputs will be dot-accessible by name
        # iterable unpacking should also give outputs
        # can assign outputs using zip(model_instance, ModelClass.output_names) ?

class DeferredType(ModelType):
    pass

class Deferred(metaclass=DeferredType):
    input = FreeField()
    output = FreeField(Direction.output)

class SubsystemSymbol(IndependentSymbol):
    # TODO: or is there a better way to check? ModelType and issubclass(Deferred)?
    model_type = DeferredType

class SubsystemField(IndependentField, symbol_class=SubsystemSymbol):
    # TODO: how to mark as a singleton?
    def __call__(self, **kwargs): # model_type
        pass

class WithDeferredSubsystems():
    """
    A decorator or mixin to mark a modelas containing a deferred subsystem

    class propulsion_interface(Deferred):
        throttle = input()
        airspeed = input()
        altitude = input()

        fuel_flow = output()
        thrust = output()


    @WithDeferredSubSystems
    class DeferredAircraft(Model):
        ...
        prop_out = propulsion_interface(throttle, airspeed, altitude)
        # prop_out.fuel_flow and prop_out.thrust are now available


    class MyEngineStatic(Model):
        paramaters_to_dynamic = ...

    class MyEngineDynamic(Model):
        parameters_from_static = ...
        throttle = ...
        airspeed = ...
        altitude = ...

    class SpecificAircraftOptimization(OptimizationProblem):
        static_engine = MyEngineStatic(

    """
    subsystem = SubsystemField()


    # TODO: all model types can get deferred systems, assume a special model (I guess it
    # wouldn't have a deferred system...) that is the only init_kwarg for 
    # DeferredSystemField. DeferredSystemModel defines IO, generally outside of
    # this model. Maybe seprately define an "InnerModel" Field for 
    # I Think it actually has to be a mixin or decorator, special check so normal model
    # construction doesn't happen until binding the deferred subsystems



class ExplicitSystem(Model):
    """
    output is an explicit function of input
    """
    # TODO: rename?
    input = FreeField()
    output = AssignedField()


class AlgebraicSystem(Model):
    """
    implicit_output are variables that drive residual to 0
    parameters are additional variables parameterizing system's explicit_output and
    residuals
    explicit_output are additional fields that depend on (solved) value of
    implicit_output and parameters
    """
    parameter = FreeField()
    #implicit_output = BoundedSymbol() # TODO: need one with bound data
    # TODO: or does this need its own field type? Or really the
    # dependent/independent is unneccessary
    implicit_output = FreeField(Direction.output)
    residual = AssignedField(Direction.internal)
    # unmatched, but maybe a subclass or imp might check lengths of residuals and
    # implicit_outputs to ensure enough DOF?
    explicit_output = AssignedField()
    initializer = MatchedField(implicit_output)


class OptimizationProblem(Model):
    """
    variable are what the optimizer moves to solve problem
    parameter are fixed wrt optimization
    constraints get assigned as relationals -- additional processing? Or should
    implementation deal with that?

    user provides objective to minimize -- if None


    """
    variable = FreeField(Direction.output)
    parameter = FreeField()
    initializer = MatchedField(variable)
    # TODO: add objective descriptor? so user code `objetive = ...` can get intercepted and
    # handled?  or it's just a contract in the API that user writes one? feasibility
    # problem if not? Add hook for metaclass to handle it?
    # needs validation for size == 1
    constraint = AssignedField(Direction.internal)
    # Can replace with "Relational" to do more validation, clear syntax candy. 
    # creaet condor type relational classes? not as pretty as using backend
    # Requires backend supports symbolic relations, but casadi and sympy do.
    # THena gain, maybe only implementation needs to deal with it
    # Does this need an "add" method, since we don't care about names, usually? Or an
    # "auto" type attrname?
    # Should we enforce that bounds on constraints are constants? probably 
    # Are constraints internal? I think so



class ODESystem(Model):
    """
    independent_variable - indepdendent variable of ODE, notionally time but can be used
    for anything. Used directly by subclasses (e.g., user code may use
    `t=DynamicsModel.independent_variable`, implementations will use this symbol
    directly for fields)

    parameter - auxilary variables (constant-in-time) that determine system behavior

    state - fully defines evolution of system driven by ODEs

    dot - derivative of state with respect to indepdnent variable, purely a function of
    state, independent_variable, parameters
    [DAE version will have a residual associated with state differential]

    output - dynamic output at particular state and value of independent variable
    (point-wise in independent variable)

    note: no time-varying input. Assume dynamicsmodels "pull" what they need from other
    models. Need to augment state with "pulled" dynamicsmodels.

    inner classes of "Event" subclass


    inner classes of "Mode" subclass



    """

    # TODO: indepdent var  needs its own descriptor type? OR do we want user classes to do
    # t = DynamicsModel.independent_variable ? That would allow leaving it like this and
    # consumers still know what it is

    # TODO: Are initial conditions an attribute of the dynamicsmodel or the trajectory 
    # analysis that consumes it?

    # TODO: mode and corresponding FiniteState type?


    independent_variable = backend.symbol_generator('t')
    state = FreeField(Direction.internal)
    # TODO: should internal free field symbolss NOT be dot accessible on the models?
    parameter = FreeField()
    dot = MatchedField(state)
    output = AssignedField()

    class Event(Model, ):
        # TODO: how to match to parent state? special __prepare__ hook on Event that DynamicsModel
        # sets up
        # update = MatchedField(state)
        # TODO: singleton event function is very similar to optimization problem objective.
        pass

