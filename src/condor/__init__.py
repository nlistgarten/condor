import numpy as np
# TODO: figure out how to make this an option/setting like django?
#from condor.backends import default as backend
from condor.fields import (
    Direction, Field, BaseSymbol, IndependentSymbol, FreeSymbol,
    IndependentField, FreeField, AssignedField, MatchedField, InitializedField,
    BoundedAssignmentField, TrajectoryOutputField,
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




# appears in __new__ as attrs
class CondorClassDict(dict):

    def __getitem__(self, *args, **kwargs):
        #breakpoint()
        return super().__getitem__(*args, **kwargs)

    def __setitem__(self, key, val):
        return super().__setitem__(key, val)
        # could be used to allow magic name update for symbols, Code below takes a
        # declared variable of the name `some_var__count__` and replaces it with 
        # `some_var_0` so you can assign variables in a loop and get unique accessors.
        # but name can get passed to FreeSymbols, and that is more explicit.
        count_suffix = '__count__'
        if key.endswith(count_suffix):
            base_name = key.split(count_suffix)[0]
            count = 0
            for ekey in self:
                if ekey.startswith(base_name):
                    count += 1
            key = key.replace(count_suffix, f"_{count}")
            breakpoint()

class Options:
    """
    Class mix-in to flag back-end options. Define an inner class on the model that
    inherits from Options that over-writes options for the backend's implementation
    (which generates a callable). Model will transparently pass attributes of Options
    subclass. Possibly only convention, but name the subclass according to the backend.
    Multiple option sub-classes provide options depending on project's default backend. 
    Single option sub-class forces back end for this model?
    OR flag, if tihs backend then these options, otherwise no options (rely on defaults)

    attribute `implementation`, if provided, should be a callable that takes a model
    and possible options and returns a callable that evaluates the model. Otherwise,
    will try to find an implementation in backend.implementations of the class name of
    the defined class or soonest MRO class name.

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

    # TODO: could have a reserved keyword "implementation" on options that defaults to
    # None, reference to an implementation to allow user provided ones? Maybe that's
    # the best way to do it, even for model types that condor ships with? if
    # inheritance works properly, can easily over-write per project, etc.




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

        for base in bases:
            _dict = base.__dict__
            for k, v in _dict.items():
                if isinstance(v, Field):
                    v_class = v.__class__
                    v_init_kwargs = v._init_kwargs.copy()
                    for init_k, init_v  in v._init_kwargs.items():
                        if isinstance(init_v, Field):
                            # TODO not sure this will be the best way to remap connected
                            # fields based on inheritance, but maybe sufficient?
                            if base.inner_to and init_v._name in base.inner_to.__dict__:
                                get_from = base.inner_to.__dict__
                            else:
                                get_from = cls_dict
                            v_init_kwargs[init_k] = get_from[init_v._name]
                    cls_dict[k] = v.inherit(
                        model_name, field_type_name=k, **v_init_kwargs
                    )
                if isinstance(v, ModelType):
                    if v.inner_to is base and v in base.inner_models:
                        # inner model inheritance & binding in new
                        cls_dict[k] = v
            if base is not Model and base.inner_to:
                for attr_name, attr_val in base.inner_to.__original_attrs__.items():
                    cls_dict[attr_name] = attr_val
                # TODO: document __from_outer__?
                cls_dict['__from_outer__'] = base.inner_to.__original_attrs__


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
        # case 1: class Model -- provides machinery to make subsequent cases easier to
        # implement.
        # case 2: ____ - library code that defines fields, etc that user code inherits
        # from (case 3+). Implementations are tied to 2? "Library Models"?
        # case 3: User Model - inherits from ____, defines the actual model that is
        # being analyzed
        # case 4: Subclass of user model to extend it?
        # I don't think InnerModel deviates from this, except perhaps disallowing
        # case 4
        # Generally, bases arg will be len <= 1. Just from the previous level. Is there
        # a case for mixins? InnerModel approach seems decent, could repeat for
        # WithDeferredSubsystems, then Model layer is quite complete 


        # TODO: thought hooks might be necessary for inner model/deferred subsystems,
        # but inner model worked well. If needed, this is a decent prototype:
        pre_super_new_hook = attrs.pop('pre_super_new_attr_hook', None)
        # pre_new_attr_hook(super_attrs, attr_name, attr_val)
        post_super_new_hook = attrs.pop('post_super_new_attr_hook', None)
        # pre_new_attr_hook(new_cls, attr_name, attr_val)

        # TODO: replace if tree with match cases?
        attrs_from_outer = attrs.pop('__from_outer__', None)
        if attrs_from_outer:
            for attr_name, attr_val in attrs_from_outer.items():
                if attrs[attr_name] is attr_val:
                    attrs.pop(attr_name)
                else:
                    # TODO: make this a warning/error? add test (really for all
                    # warning/error raise)
                    print(f"{attr_name} was defined on outer of {name}")

        print("CondorModelType.__new__ for", name, bases, kwargs)
        print(attrs)

        # perform as much processing as possible before caller super().__new__
        # by building up super_attrs from attrs; some operations need to operate on the
        # constructed class, like binding fields and innermodels

        super_attrs = {}

        if bases:
            super_attrs['_parent_name'] = bases[0].__name__
            if 'inner_to' not in kwargs and bases[0] is not Model and bases[0].inner_to:
                kwargs['inner_to'] = bases[0].inner_to
        else:
            super_attrs['_parent_name'] = ''

        backend_options = {}
        independent_fields = [] # used to find free symbols and replace with symbol
        matched_fields = []

        # will be used to pack arguments in (inputs) and unpack results (output), other
        # convenience
        input_fields = []
        output_fields = []
        internal_fields = []
        input_names = []
        output_names = []
        inner_models = []

        # TODO: better reserve word check? see check attr name below
        orig_attrs = {k:v for k,v in attrs.items() if not k.startswith("__")}

        # TODO: use a special inner class like _meta to de-clutter model namespace?
        super_attrs.update(dict(
            input_fields = input_fields,
            output_fields = output_fields,
            input_names = input_names,
            output_names = output_names,
            inner_models = [],
            __original_attrs__ = orig_attrs,
        ))


        for attr_name, attr_val in attrs.items():
            # TODO: process name clahees on attr_name? can't protect against internal
            # over-write (maybe user intends to do that) but can raise error for
            # over-writing parent, e.g., creating a "state" variable in a DynamicModel
            # subclass. Implemented some name clash checks, but can it be easier by
            # changing the condor dict setattr?

            # need to have protected names? I guess whatever we're doing for Options and
            # convenience fields like input_fields, input_fields, etc?

            pass_attr = True

            if isinstance(attr_val, type) and issubclass(attr_val, Options):
                backend_options[attr_name] =  attr_val
                continue
            if isinstance(attr_val, IndependentField):
                independent_fields.append(attr_val)
            if isinstance(attr_val, MatchedField):
                matched_fields.append(attr_val)
            if isinstance(attr_val, Field):
                if attr_val._direction == Direction.input:
                    input_fields.append(attr_val)
                if attr_val._direction == Direction.output:
                    output_fields.append(attr_val)
                if attr_val._direction == Direction.internal:
                    internal_fields.append(attr_val)
            if isinstance(attr_val, backend.symbol_class):
                # from a FreeField
                known_symbol_type = False
                for free_field in independent_fields:
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

                # TODO: MatchedField in "free" mode
                # TODO: from the output of a subsystem? Does this case matter? 

                if not known_symbol_type:
                    print("unknown symbol type", attr_name, attr_val)
                    # Hit by DynamicModel.independent_variable

            # TODO: other possible attr types to process:
            # maybe field dataclass, if we care about intervening in eventual assignment
            # deferred subsystems, but perhaps this will never get hit? need to figure
            # out how a model with deferred subsystem behaves

            if isinstance(attr_val, ModelType):
                if attr_val.inner_to in bases and attr_val in attr_val.inner_to.inner_models:
                    # handle inner classes below, don't add to super
                    continue

            if pass_attr:
                check_attr_name(attr_name, attr_val, super_attrs, bases)
                super_attrs[attr_name] = attr_val

        # before calling super new, process fields

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

        inner_to=kwargs.pop('inner_to', None)
        inner_through=kwargs.pop('inner_through', None)

        new_cls = super().__new__(cls, name, bases, super_attrs, **kwargs)

        # handle inner classes
        new_cls.inner_to = inner_to
        if inner_to:
            # TODO: other validation?
            if new_cls.__name__ in inner_to.__dict__:
                raise ValueError

            subclass_of_inner = False
            for base in new_cls.__bases__:
                if base is not Model:
                    if base in inner_to.inner_models:
                        subclass_of_inner = True
                        break
                        # TODO: base is ~ the field that cls should be added to...
                        # This could all be in ModelType.__new__, inner_to is in kwargs

            if subclass_of_inner:
                # register as symbol of field, not directly added
                if not inner_through:
                    raise ValueError
            else:
                # create as field
                inner_to.inner_models.append(new_cls)
                setattr(inner_to, new_cls.__name__, new_cls)

        if inner_through:
            # create links
            inner_through.register(new_cls)
            new_cls.inner_through = inner_through
            new_cls.inner_to = inner_through.inner_to

        for attr_name, attr_val in attrs.items():
            if isinstance(attr_val, Field):
                attr_val.bind(attr_name, new_cls)

            if isinstance(attr_val, ModelType):
                if attr_val.inner_to in bases and attr_val in attr_val.inner_to.inner_models:
                    # attr_val is an inner model to base, so this is a field-like inner
                    # model that the user will sub-class

                    # new_cls is a user model that inherits an inner model type

                    use_bases = (InnerModel,)
                    if Model in attr_val.__bases__ and len(attr_val.__bases__) == 1:
                        pass
                    else:
                        raise NotImplemented
                        # does this work?
                        use_bases = use_bases + attr_val.__bases__

                    attr_val = InnerModelType(
                        attr_name,
                        use_bases,
                        attr_val.__original_attrs__,
                        inner_to = new_cls,
                        original_class = attr_val,
                    )
                    setattr(new_cls, attr_name, attr_val)


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
            # TODO: search MRO?
            for base in bases:
                if (
                    isinstance(base, ModelType) and 
                    (base.__name__ in backend.implementations.__dict__)
                ):
                    # TODO: or are these directly on backend along with symbol_class,
                    # symbol_generator, and other things are in a utils submodule?
                    implementation = getattr(
                        backend.implementations,
                        base.__name__
                    )
                    break

        if implementation is not None:
            new_cls.implementation = implementation(new_cls, **backend_option)

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
    """
    Define a ____ by subclassing Model, creating field types, and writing an
    implementation.
    """
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
        self.bind_input_fields()

        cls.implementation(self, *list(input_kwargs.values()))

        # generally implementations are responsible for binding computed values.
        # implementations know about models, models don't know about implementations
        self.output_kwargs = output_kwargs = {
            out_name: getattr(self, out_name)
            for field in self.output_fields
            for out_name in field.list_of('name')
        }

    def bind_input_fields(self):
        cls = self.__class__
        all_values = list(self.input_kwargs.values())

        slice_start = 0
        for field in cls.input_fields:
            slice_end = slice_start + len(field)
            values = all_values[slice_start: slice_end]
            slice_start = slice_end
            self.bind_field(field, values, symbols_to_instance=True, wrap=False)


    def bind_field(self, field, values, symbols_to_instance=True, wrap=True):
        dataclass_kwarg = {}
        if wrap:
            values = backend.wrap(field, values)
        for symbol_name, value in zip(field.list_of('name'), values):
            dataclass_kwarg[symbol_name] = value
            if symbols_to_instance:
                setattr(self, symbol_name, value)
        setattr(self, field._name, field._dataclass(**dataclass_kwarg))

    def __iter__(self):
        cls = self.__class__
        for output_name in cls.output_names:
            yield getattr(self, output_name)

    def __repr__(self):
        return f"<{self.__class__.__name__}: " + ", ".join([f"{k}={v}" for k, v in self.input_kwargs.items()]) + ">"


class InnerModelType(ModelType):
    def __iter__(cls):
        for subclass in cls.subclasses:
            yield subclass

    def register(cls, subclass):
        cls.subclasses.append(subclass)

    def __new__(cls, name, bases, attrs, inner_to=None, original_class = None,  **kwargs):
        print("\nInnerModelType.__new__ for class", cls,"name", name, "bases", bases,
              "original", original_class, "\nkwargs:", kwargs, "\nattrs:", attrs, "\n")
        # case 1: InnerModel
        # case 2: library inner model inherited to user model through user model's __new__
        # case 3: subclass of inherited inner model -- user
        case1 = name == "InnerModel"
        case2 = inner_to is not None and original_class is not None
        case3 = not case1 and not case2
        if case1 or case2:
            new_cls = super().__new__(cls, name, bases, attrs, inner_to=inner_to, **kwargs)

        if case2:
            new_cls.original_class = original_class
            new_cls.subclasses = []
            # reference to original class so uer sub-classes inherit directly, see below

        if case3:
            if len(bases) > 1:
                raise ValueError
            inner_through = bases[0]
            original_class = inner_through.original_class
            # subclass original class, inheriting
            # will inherit inner_to from original_class
            new_cls = original_class.__class__(name, (original_class,), attrs,
                                               inner_through=inner_through, **kwargs)

        return new_cls


class InnerModel(Model, metaclass=InnerModelType, inner_to=None):
    """
    Create an inner model ____ by assigning an inner_to keyward arguement to a ____. The
    argument references another ____. Then a user outer model (a subclass of ____)
    can create multiple subclass of the InnerModel by sub-classing from
    <outer_model>.<inner_model ____>
    """
    pass


# TODO: move deferred stuff out?
class DeferredType(ModelType):
    """
    Deferred model's are of type DeferredType
    """
    pass

class Deferred(Model, metaclass=DeferredType):
    """
    Deferred model's have only model input/output defined
    """
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


# TODO: Move to "contrib" or something?
class ExplicitSystem(Model):
    """
    output is an explicit function of input
    """
    input = FreeField()
    output = AssignedField()

class Tablelookup(Model):
    """
    rectilnear -> structured, rectilinear grid
    later, unstructured lookup.

    inputs define input variables
    outputs are names of output

    knots are vector of grid points for each input
    data is grid of data for each output of shape generated by meshgrid ij of all knots

    options:
    free or fixed?
    spline degree

    """

    input = FreeField()
    output = FreeField()

    knots = MatchedField(input) # vector of grid points for each input
    data = MatchedField(output) # grid of data for each output of shape generated by


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
    constraint = BoundedAssignmentField(Direction.internal)


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
    But do need a "control" that could be defined based on mode? 

    is "mode" the only finite state?
    mode can be an innner model that assigns controls (direct over-write should be
    possible)
    I suppose you could have different modes that are used to define independent
    controls. OK. 

    inner model "Event" (see below). For a MyODESyStem model, create a sub-class of
    MyODESystem.Event,
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
    initial = MatchedField(state)
    # finite state is an idiom not a unique field
    # finite_state = FreeField(Direction.internal)
    # TODO: should internal free field symbolss NOT be dot accessible on the models?
    parameter = FreeField()
    dot = MatchedField(state)
    control = FreeField(Direction.internal)
    output = AssignedField()

# TODO: need to exlcude fields, particularly dot, initial, etc. 
# define which fields get lifted completely, which become "read only" (can't generate
# new state) etc. 
class Event(Model, inner_to = ODESystem):
    # TODO: singleton field event.function is very similar to objective in
    # OptimizationProblem
    update = MatchedField(ODESystem.state, direction=Direction.output)
    # make[mode_var] = SomeModeSubclass
    # actually, just update it
    #make = MatchedField(ODESystem.finite_state)

class Mode(Model, inner_to=ODESystem):
    make = MatchedField(ODESystem.control)
    # e.g., make[alpha] = ...
    # only for control signals? and any control signals must be set by mode?
    # maybe also provide different dot override?
    # condition attribute is a symbolic relation used to determine when make is applied



# TODO: trajectory should ex
# TODO: decide on better model API for how state, etc. interact between ODESystem and
# trajectory analysis. Will impact Implementation and TrajectoryOutputFIeld
class TrajectoryAnalysis(Model, inner_to=ODESystem):
    trajectory_output = TrajectoryOutputField()
    # singleton simulate_to? or t_f?


