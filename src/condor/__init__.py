import numpy as np
# TODO: figure out how to make this an option/setting like django?
#from condor.backends import default as backend
from condor.fields import (
    Direction, Field, BaseSymbol, IndependentSymbol, FreeSymbol,
    IndependentField, FreeField, AssignedField, MatchedField, InitializedField
)
from condor.backends.default import backend
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
    # TODO: do meta programming so implementation enums are available without import?




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
        return super().__call__(*args, **kwargs)

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
                check_attr_name(out_name, out_symbol, super_attrs, bases)
                super_attrs[out_name] = out_symbol
                output_names.append(out_name)
            output_field.create_dataclass()

        # process docstring
        lhs_doc = ', '.join([out_name for out_name in output_names])
        arg_doc = ', '.join([arg_name for arg_name in input_names])
        orig_doc = attrs.get("__doc__", "")
        super_attrs["__doc__"] = "\n".join([orig_doc, f"    {lhs_doc} = {name}({arg_doc})"])

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

    if attr_name in ['__module__', '__qualname__', '__doc__']:
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
        # TODO: is there a better way to do this?
        input_kwargs = {}
        for input_name, input_val in zip(cls.input_names, args):
            if input_name in kwargs:
                raise ValueError
            input_kwargs[input_name] = input_val
        input_kwargs.update(kwargs)
        self.input_kwargs = input_kwargs

        # TODO: check bounds on model inputs?

        # pack into dot-able storage, over-writting fields and symbols
        for input_field in cls.input_fields:
            dataclass_kwarg = {}
            for input_name in input_field.list_of('name'):
                dataclass_kwarg[input_name] = input_kwargs[input_name]
                setattr(self, input_name, input_kwargs[input_name])
            setattr(self, input_field._name, input_field._dataclass(**dataclass_kwarg))

        imp_out = cls.implementation(*list(input_kwargs.values()))

        self.output_kwargs = output_kwargs = {
            out_name: val for out_name, val in zip(cls.output_names, imp_out)
        }

        # pack into dot-able storage, over-writting fields and symbols
        for output_field in cls.output_fields:
            dataclass_kwarg = {}
            for output_name in output_field.list_of('name'):
                dataclass_kwarg[output_name] = output_kwargs[output_name]
                setattr(self, output_name, output_kwargs[output_name])
            setattr(self, output_field._name, output_field._dataclass(**dataclass_kwarg))

    def __iter__(self):
        cls = self.__class__
        for output_name in cls.output_names:
            yield getattr(self, output_name)

    def __repr__(self):
        return f"<{self.__class__.__name__}: " + ", ".join([f"{k}={v}" for k, v in self.input_kwargs.items()]) + ">"


# TODO: move deferred stuff out?
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


# Move to "contrib" or something?
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
    implicit_output = InitializedField(Direction.output)
    residual = AssignedField(Direction.internal)
    # unmatched, but maybe a subclass or imp might check lengths of residuals and
    # implicit_outputs to ensure enough DOF?
    explicit_output = AssignedField()


class OptimizationProblem(Model):
    """
    variable are what the optimizer moves to solve problem
    parameter are fixed wrt optimization
    constraints get assigned as relationals -- additional processing? Or should
    implementation deal with that?

    user provides objective to minimize -- if None


    """
    variable = InitializedField(Direction.output)
    parameter = FreeField()
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

