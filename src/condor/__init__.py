import numpy as np
# TODO: figure out how to make this an option/setting like django?
#from condor.backends import default as backend
from condor.fields import (
    Direction, Field, BaseSymbol, IndependentSymbol, FreeSymbol, WithDefaultField,
    IndependentField, FreeField, AssignedField, MatchedField, InitializedField,
    BoundedAssignmentField, TrajectoryOutputField,
)
from condor.backends.default import backend
from condor.conf import settings
from dataclasses import asdict
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

# TODO: a way to update options
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

    # TODO: allow a generic options for non-solver specific options?


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
    # think definitely, and probably can't protect user model from over-writing self in
    # model but can protect by modifying setattr on CondorDict


    @classmethod
    def __prepare__(cls, model_name, bases, **kwds):
        print("CondorModelType.__prepare__  for", model_name)
        sup_dict = super().__prepare__(cls, model_name, bases, **kwds)
        cls_dict = CondorClassDict(**sup_dict)

        # TODO: may need to search MRO resolution, not just bases, which without mixins
        # are just singletons. For fields and inner classes, since each generation of
        # class is getting re-inherited, this is sufficient. 
        for base in bases:
            _dict = base.__dict__
            for k, v in _dict.items():
                # inherit fields from base -- bound in __new__
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
                # inherit inner model from base, create reference now, bind in new
                elif isinstance(v, ModelType):
                    if v.inner_to is base and v in base.inner_models:
                        # inner model inheritance & binding in new
                        cls_dict[k] = v
                elif isinstance(v, backend.symbol_class):
                    cls_dict[k] = v

            # if base is an inner model, make reference  of outer model attributes
            # and copy fields from class __copy_fields__ kwarg definition. This allows
            # inner field to modify local copy without affecting outer model
            if base is not Model and base.inner_to:
                for attr_name, attr_val in base.inner_to.__original_attrs__.items():
                    # TODO: document copy fields? can this get DRY'd up?
                    # don't like that symbol copying is here, maybe should be method on
                    # field?
                    if attr_name in base.__copy_fields__:
                        field_class = attr_val.__class__
                        field_init_kwargs = attr_val._init_kwargs.copy()
                        # I know it has no field references
                        copied_field = attr_val.inherit(
                            model_name, field_type_name=attr_name, **field_init_kwargs
                        )
                        copied_field._symbols = [sym for sym in attr_val]
                        copied_field._count += sum(copied_field.list_of('size'))
                        cls_dict[attr_name] = copied_field
                        continue

                    cls_dict[attr_name] = attr_val
                # TODO: document __from_outer__?
                cls_dict['__from_outer__'] = base.inner_to.__original_attrs__
        # TODO: is it better to do inner model magic in InnerModelType.__prepare__?


        print("end prepare for", model_name,)# cls_dict)
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
        # maybe call this model template?
        # from (case 3+). Implementations are tied to 2? "Library Models"?
        # case 3: User Model - inherits from ____, defines the actual model that is
        # being analyzed
        # case 4: Subclass of user model to extend it?
        # I don't think InnerModel deviates from this, except perhaps disallowing
        # case 4
        # Generally, bases arg will be len <= 1. Just from the previous level. Is there
        # a case for mixins? InnerModel approach seems decent, could repeat for
        # WithDeferredSubsystems, then Model layer is quite complete 

        # TODO: add support for inheriting other models -- subclasses are
        # modifying/adding (no deletion? need mixins to pre-build pieces?) to parent
        # classes.  case 4
        # fields would need to combine _symbols


        # TODO: thought hooks might be necessary for inner model/deferred subsystems,
        # but inner model worked well. If needed, this is a decent prototype:
        # ACTUALLY -- subclass of ModelType should be used, can do things before/after
        # calling super new?
        pre_super_new_hook = attrs.pop('pre_super_new_attr_hook', None)
        # pre_new_attr_hook(super_attrs, attr_name, attr_val)
        post_super_new_hook = attrs.pop('post_super_new_attr_hook', None)
        # pre_new_attr_hook(new_cls, attr_name, attr_val)


        # __from_outer__ attribute is attached to InnerModel`s during __prepare__ to
        # give references to IndependentVariable backend_repr`s conveniently for
        # constructing InnerModel symbols. They get cleaned up since they live in the
        # outer model
        attrs_from_outer = attrs.pop('__from_outer__', {})
        for attr_name, attr_val in attrs_from_outer.items():
            if attrs[attr_name] is attr_val:
                attrs.pop(attr_name)
            else:
                # TODO: make this a warning/error? add test (really for all
                # warning/error raise)
                # Or allow silent passage of redone variables? eg copying
                # but copying is a special case that can be guarded...
                print(f"{attr_name} was defined on outer of {name}")

        print("CondorModelType.__new__ for", name, bases, kwargs)



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

        inner_to=kwargs.pop('inner_to', None)
        inner_through=kwargs.pop('inner_through', None)
        copy_fields=kwargs.pop('copy_fields', [])

        backend_options = {}
        matched_fields = []

        # used to find free symbols and replace with symbol, seeded with outer model
        independent_fields = [
            v for k, v in attrs_from_outer.items()
            if isinstance(v, IndependentField) and k not in copy_fields
        ]


        # will be used to pack arguments in (inputs) and unpack results (output), other
        # convenience
        input_fields = []
        output_fields = []
        internal_fields = []
        input_names = []
        output_names = []
        inner_models = []
        sub_models = {}

        # TODO: better reserve word check? see check attr name below
        orig_attrs = {k:v for k,v in attrs.items() if not k.startswith("__")}

        # TODO: use a special inner class like _meta to de-clutter model namespace?
        super_attrs.update(dict(
            input_fields = input_fields,
            output_fields = output_fields,
            input_names = input_names,
            output_names = output_names,
            internal_fields = internal_fields,
            independent_fields = independent_fields,
            inner_models = [],
            sub_models = sub_models,
            __original_attrs__ = orig_attrs,
        ))


        # TODO: replace if tree with match cases?
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
            if isinstance(attr_val.__class__, cls):
                sub_models[attr_name] = attr_val
            if isinstance(attr_val, backend.symbol_class):
                # from a IndependentField
                known_symbol_type = False
                for free_field in independent_fields:
                    symbol = free_field.get(backend_repr=attr_val)
                    # not a list (empty or len > 1)
                    if isinstance(symbol, BaseSymbol):
                        known_symbol_type = True
                        if symbol.name and symbol.name != attr_name:
                            raise NameError(f"Symbol on {free_field} has name {symbol.name} but assigned to {attr_name}")
                        if attr_name:
                            symbol.name = attr_name
                        else:
                            symbol.name = f"{field._model_name}_{field._name}_{field._symbols.index(symbol)}"
                        attr_val = symbol
                        # pass attr if field is bound (_model is a constructed Model
                        # class, not None), otherwise will get added later after more
                        # processing
                        pass_attr = free_field._model is not None
                        if pass_attr:
                            # TODO is this kind of defensive checking useful?
                            assert issubclass(free_field._model, inner_to)
                        break

                # TODO: MatchedField in "free" mode
                # TODO: from the output of a subsystem? Does this case matter?

                if not known_symbol_type:
                    print("unknown symbol type", attr_name)#, attr_val)
                    #sub_models[attr_name] = attr_val
                    # Hit by ODESystem.t, is that okay?
                    # TODO: maybe DONT pass these on. they don't work to use in the
                    # another model since they don't get bound correctly, then just have
                    # dangling casadi expression/symbol. Event functions can just use
                    # ODESystem.t, or maybe handle as a special case? Could  check to
                    # see if it's in the template, and only keep those?

            # TODO: other possible attr types to process:
            # maybe field dataclass, if we care about intervening in eventual assignment
            # deferred subsystems, but perhaps this will never get hit? need to figure
            # out how a model with deferred subsystem behaves

            if isinstance(attr_val, ModelType):
                if attr_val.inner_to in bases and attr_val in attr_val.inner_to.inner_models:
                    # handle inner classes below, don't add to super
                    continue

            # Options, InnerModel, and IndependentSymbol are not added here
            if pass_attr:
                check_attr_name(attr_name, attr_val, super_attrs, bases)
                super_attrs[attr_name] = attr_val

        # before calling super new, process fields

        for field in independent_fields:
            for symbol in field:
                # add names to symbols -- must be an unnamed symbol without a reference
                # assignment in the class
                if not symbol.name:
                    symbol.name = f"{field._model_name}_{field._name}_{field._symbols.index(symbol)}"
                    print("setting name for", symbol.name)

        for matched_field in matched_fields:
            for matched_symbol in matched_field:
                matched_symbol.update_name()

        # symbols from input and input fields are added directly to model
        # previously, all fields were  "finalized" by creating dataclass
        # ODESystem has no implementation and not all the fields should be finalized 
        # Events may create new parameters that are canonical to the ODESystem model,
        # and should be not be finalized until a trajectory analysis, which has its own
        # copy. state must be fully defined by ODEsystem and can/should be finalized.
        # What about output? Should fields have an optional skip_finalize or something
        # that for when an inner class may modify it so don't finalize? And is that the
        # only time it would come up?
        # TODO: review this 

        for input_field in input_fields:
            for in_symbol in input_field:
                in_name = in_symbol.name
                check_attr_name(in_name, in_symbol, super_attrs, bases)
                super_attrs[in_name] = in_symbol
                input_names.append(in_name)

        for internal_field in internal_fields:
            internal_field.create_dataclass()

        for output_field in output_fields:
            for out_symbol in output_field:
                out_name = out_symbol.name
                check_attr_name(out_name, out_symbol, super_attrs, bases)
                super_attrs[out_name] = out_symbol
                output_names.append(out_name)
            output_field.create_dataclass()

        # process docstring / add simple equation
        lhs_doc = ', '.join([out_name for out_name in output_names])
        arg_doc = ', '.join([arg_name for arg_name in input_names])
        orig_doc = attrs.get("__doc__", "")
        super_attrs["__doc__"] = "\n".join([orig_doc, f"    {lhs_doc} = {name}({arg_doc})"])


        new_cls = super().__new__(cls, name, bases, super_attrs, **kwargs)

        # creating an InnerClass
        # inner_to kwarg is added to InnerModel  template definition or for subclasses
        # of innermodel templates, in InnerModel.__new__. For subclasses of template,
        # reference to base is inner_through
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
                new_cls.__copy_fields__ = copy_fields

        if inner_through:
            # create links
            inner_through.register(new_cls)
            new_cls.inner_through = inner_through
            new_cls.inner_to = inner_through.inner_to

        # Bind Fields and InnerModels -- require reference to constructed Model
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

                    # TODO: can we dry up all these extra args? or is it necessary to
                    # specify these? or at leat document that this is where new features
                    # on inner models need to be added, in addition to poping from
                    # kwargs, etc. When reverting inner model stuff to InnerModel's init
                    # subclass, may be more obvious

                    attr_val = InnerModelType(
                        attr_name,
                        use_bases,
                        attr_val.__original_attrs__,
                        inner_to = new_cls,
                        original_class = attr_val,
                        copy_fields = attr_val.__copy_fields__,
                    )
                    setattr(new_cls, attr_name, attr_val)


        # Bind implementation
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
            cls.finalize_input_fields(new_cls)
            new_cls.implementation = implementation(new_cls, **backend_option)

        return new_cls

    def finalize_input_fields(cls):
        for input_field in cls.input_fields:
            input_field.create_dataclass()


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

    if getattr(attr_val, 'backend_repr', None) == super_attrs.get(attr_name, None):
        return

    # TODO: if user-defined field starts with _, raise value error?

    in_bases = False
    for base in bases:
        # TODO: only need to check the parent, not all bases?
        if (
            attr_name in base.__dict__ and
            not (
                getattr(attr_val, '_inherits_from', None) == getattr(base, attr_name)
                or attr_val == getattr(base, attr_name)
            )
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
    Define a Model Template  by subclassing Model, creating field types, and writing an
    implementation.
    """
    def __init__(self, *args, **kwargs):
        cls = self.__class__
        # bind *args and **kwargs to to appropriate signature
        # TODO: is there a better way to do this?
        input_kwargs = {}
        for input_name, input_val in zip(cls.input_names, args):
            if input_name in kwargs:
                raise ValueError(f"Argument {input_name} has value {input_val} from args and {kwargs[input_name]} from kwargs")
            input_kwargs[input_name] = input_val
        input_kwargs.update(kwargs)
        try:
            self.input_kwargs = {name: input_kwargs[name] for name in cls.input_names}
        except KeyError as e:
            raise ValueError(f"Could not find keyword arguments {e.args} in {cls.__name__}")
        for key in kwargs:
            if key not in self.input_kwargs:
                raise ValueError(f"Unexpected keyword argument {key} in {cls.__name__}")

        # TODO: check bounds on model inputs?

        # pack into dot-able storage, over-writting fields and symbols
        self.bind_input_fields()

        cls.implementation(self, *list(self.input_kwargs.values()))

        # generally implementations are responsible for binding computed values.
        # implementations know about models, models don't know about implementations
        self.output_kwargs = output_kwargs = {
            out_name: getattr(self, out_name)
            for field in self.output_fields
            for out_name in field.list_of('name')
        }

        self.bind_submodels()


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

    def bind_submodels(model_instance):
        # TODO: how to have models cache previous results so this is always free?
        # Can imagine a parent model with multiple instances of the exact same
        # sub-model called with different parameters. Would need to memoize at least
        # that many calls, possibly more.
        model = model_instance.__class__
        print(f"binding sub-models on {model}")
        model_assignments = {}

        fields = [
            field
            for field in model.input_fields + model.output_fields
            if isinstance(field, IndependentField)
        ]
        for field in fields:
            model_instance_field = getattr(model_instance, field._name)
            model_instance_field_dict = asdict(model_instance_field)
            model_assignments.update({
                elem.backend_repr: val
                for elem, val in zip(field, model_instance_field_dict.values())
            })

        for sub_model_ref_name, sub_model_instance in model.sub_models.items():
            sub_model = sub_model_instance.__class__
            print(f"binding sub-model {sub_model} on {model}")
            sub_model_kwargs = {}

            for field in sub_model.input_fields:
                bound_field = getattr(sub_model_instance, field._name)
                bound_field_dict = asdict(bound_field)
                for k,v in bound_field_dict.items():
                    if not isinstance(v, backend.symbol_class):
                        sub_model_kwargs[k] = v
                    else:
                        value_found = False
                        for kk, vv in model_assignments.items():
                            if backend.utils.symbol_is(v, kk):
                                value_found = True
                                break
                        sub_model_kwargs[k] = vv
                        if not value_found:
                            # TODO: this is the only casadi specific thing :)
                            sub_model_kwargs[k] = backend.utils.evalf(
                                v, model_assignments
                            )

            bound_sub_model = sub_model(**sub_model_kwargs)
            setattr(model_instance, sub_model_ref_name, bound_sub_model)
            #bound_sub_model.recursive_bind()





class InnerModelType(ModelType):
    def __iter__(cls):
        for subclass in cls.subclasses:
            yield subclass

    def register(cls, subclass):
        cls.subclasses.append(subclass)

    def __new__(cls, name, bases, attrs, inner_to=None, original_class = None,  **kwargs):
        print("\nInnerModelType.__new__ for class", cls,"name", name, "bases", bases,
              "original", original_class, "\nkwargs:", kwargs,"\n")# "\nattrs:", attrs, "\n")
        # case 1: InnerModel definition
        # case 2: library inner model inherited to user model through user model's __new__
        # (inner model template)
        # case 3: subclass of inherited inner model -- user defined model
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
    Create an inner model template  by assigning an inner_to keyward arguement during
    model template definition. The argument references another model template (which
    becomes the "outer model"). Then a user outer model (a subclass of the inner model
    template which is bound to outer model) can create multiple subclass of the 
    InnerModel by sub-classing from <outer_model>.<inner_model template>
    """
    pass

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

    detect if knots and/or data are symbolic or numeric. If symbolic, treat as input

    options:
    free or fixed?
    spline degree

    """

    # TODO enforce shape = 1 for input/output??

    input = FreeField()
    output = FreeField(Direction.output)

    input_data = MatchedField(input) # vector of grid points for each input
    output_data = MatchedField(output) # grid of data for each output of shape generated by


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
    t - indepdendent variable of ODE, notionally time but can be used
    for anything. Used directly by subclasses (e.g., user code may use
    `u=DynamicsModel.t`, implementations will use this symbol
    directly for fields)

    parameter - auxilary variables (constant-in-time) that determine system behavior

    state - fully defines evolution of system driven by ODEs

    dot - derivative of state with respect to indepdnent variable, purely a function of
    state, independent_variable, parameters
    [DAE version will have a residual associated with state differential]

    output - dynamic output at particular state and value of independent variable
    (point-wise in independent variable)

    control - time-varying placeholder, useful for re-defining using mode, etc.
    set it with `make` field -- all controls MUST be set? or default to 0.

    make - set control a value/computation. Or `let`?

    note: no time-varying input to system. Assume dynamicsmodels "pull" what they need 
    from other models. 
    Need to augment state with "pulled" dynamicsmodels
    But do need a "control" that could be defined based on mode? automatically added to
    output? or is it always just a placeholder for functional space gradient?
    Or also allow it to define eg feedback control

    I guess block diagram is nice for simulating something like saturation block --
    create event and switch modes? but I guess that's re-creatable with modes. But maybe
    needs control/subsitution -- I guess what I'm calling "control" is really an
    (explicit) algebraic state? I guess this is really the same as an "output" and is in
    fact how simupy implements it. maybe convert output to a freefield that takes an
    expression (like constraint, I guess?) and then "make" is only in mode and adds
    conditional behavior -- defaults to value from creation?

    then keep control field but only for open loop control? not sure, but some way to
    mark an output as a control signal

    don't include specialty types like discrete control, etc? user must know that for
    discrete time signal it requires state with dot=0 and update

    dt - reserved keyword for DT?

    inner model "Event" (see below). For a MyODESyStem model, create a sub-class of
    MyODESystem.Event,
    inner classes of "Mode" subclass -- conditional overwrite of dot and make
    inner classes of TrajectoryAnalysis -- perform simulations with particular
    trajectory outputs. Local copy of parameters, initial to allow sim-specific



    """

    # TODO: indepdent var  needs its own descriptor type? OR do we want user classes to do
    # t = DynamicsModel.independent_variable ? That would allow leaving it like this and
    # consumers still know what it is
    # or just set it to t and always use it? trying this way...

    # TODO: Are initial conditions an attribute of the dynamicsmodel or the trajectory 
    # analysis that consumes it?
    # Currently, ODESystem sets a default that TrajectoryAnalysis can override for
    # current sim. I think I like this

    # TODO: mode and corresponding FiniteState type?
    # Yes mode with condition, no FiniteState -- this is an idiom for a state with no
    # dot and only updated to integer values. Expect to use in conditions with ==. Can
    # python enum values get passed to backend?

    # TODO: convenience for setting max_step based on discrete time systems? And would
    # like event functions of form (t - te) to loop with a specified tf. For now can set
    # max_step in trajectoryanalysis options manually... OR figure out how to specify
    # periodic exact-time simulation and then current machinery should be fine

    # TODO: decide how controls work :) Ideally, easy to switch between continuous and
    # sampled feedback, open-loop control. Open-loop would need some type of
    # specification so adjoint gradients could get computed??

    # TODO: combine all events if functions are the same? maybe that happens
    # automatically and that's okay. Needs the new events API (mimicking sundials) to happen
    # --> VERIFY that SGM works correctly? Would require non-conflicting update
    # functions

    # TODO: inside another model, the callback gets captured into the "expression" and
    # will never hit the model's or implementation's __call__ methods. Would really like
    # to 1) bind the simulation result (as in the implementation call) and/or 2)
    # initialize the model since that is most likely where we would hook the
    # sweepable-like logging. This is also/really a question about how to inject Model
    # layer back into casadi expressions. Could be used for capturing dangling
    # computations which currently leaning away from doing but trajectory is a special
    # case... If the trajectory analysis only appears once in the other model could use
    # the caching of result object on callback to 
    # Can set an attribute (`from_implementation`) on callback  assuming we make sure
    # new  instances of implementation (and callback) are created to ensure re-entrant.
    # Then refactor binding to create new model instance as appropriate. This would be
    # version #2, probably can't figure out what model called the trajectory analysis
    # and bind in which is probably more consistent anyway.

    # TODO: does the simulation result need to be included in the template somehow?
    # Or is this an implementation detail and current approach is fine? Or how to
    # indicate that time, state, and output fields are time varying and should be
    # written as an arrray and not to eg the database?

    # SimuPy-coupled TODO

    # TODO switch to scikits.odes, especially updating update function API to get an
    # array of length num_events w/ elements 0 for did not occur and +/-1 for direction
    # don't use nan's to terminate? althogh, can't code-gen termination?

    # --> MAKE SURE scikits.odes is re-entrant. AND important TODO: figure out how to
    # create new implementation instances as needed for parallelization. Not sure if
    # that would happen automatically with pickle, etc.

    # TODO: simupy currently requires dim_output > 0, which kinda makes sense for the
    # BlockDiagram case and kinda makes sense for disciplined dynamical system modeling,
    # but maybe doesn't make sense for single system simulation?
    # IF we were keeping intermediate computations, they would be good candidates

    # OR can/should trajectory output integrandds get added to simupy outputs, save a
    # for-loop?

    # what is a good output for adjoint system? gradient integrand term!

    # TODO: currently, to get a discrete time control need to augment state and provide
    # a separate initializer, even though it's  generally going to be the same
    # expression as the update. Should that be fixed in simupy? event funciton = 0 -> do
    # update? I can fix it in casadi shooting_gradient_method.py as well.

    # or is initial where we want it? consistent with initial conition processing for
    # SGM. 

    # Are simupy outputs even good? need to be disciplined to keep system encapsolation
    # anyway, and control as discrete state prevents the re-computation. I guess
    # back-to-back computation of a "continuous" optimization (not sure if DAEs can do
    # this) is fine too? I think DAE version will be even more efficient, since
    # everything is a "state" and computed simultaneously. Can you do MAUD approach in
    # IDA to provide solvers for explicit expressions? 

    # TODO: add event times/event channels to SimulationResult during detection

    # TODO: don't like hacks to simupy to make it work... especially the success
    # checking stuff -- is that neccessary anymore?

    t = backend.symbol_generator('t')
    state = FreeField(Direction.internal)
    initial = MatchedField(state)
    parameter = FreeField()
    dot = MatchedField(state)
    modal = WithDefaultField(Direction.internal)
    dynamic_output = AssignedField(Direction.internal)


class TrajectoryAnalysis(
        Model,
        inner_to=ODESystem,
        copy_fields=["parameter", "initial", "state", "dynamic_output"]
):
    """
    this is what simulates an ODE system
    tf parameter 
    define trajectory outputs
    modify parameters and initial (local copy)

    gets _res, t, state, output (from odesystem) assigned from simulation

    """
    trajectory_output = TrajectoryOutputField()
    default_tf = 1_000_000.

    # TODO: how to make trajectory outputs that depend on other state's outputs without
    # adding an accumulator state and adding the updates to each event? Maybe that
    # doesn't make sense...

# TODO: need to exlcude fields, particularly dot, initial, etc.
# define which fields get lifted completely, which become "read only" (can't generate
# new state) etc.
# maybe allow creation of new parameters (into ODESystem parameter field), access to
# state, etc.
class Event(Model, inner_to = ODESystem):
    """
    update for any state that needs it

    terminate is a boolean flag to indicate whether the event terminates the simulation
    (default is False)

    function defines when event occurs OR at_time is an expression or slice of
    expressions for when the event occurs. Slice assumes constant dt between start and
    stop. use stop=None for infinite oscillator. Differs from standard slice
    semantics in that both start and stop are inclusive, so slice(0, 1, 1) will generatw
    two events at 0 and 1. This means an at_time of a single value of t_e is equiavelent
    to slice(t_e, t_e, None)

    """
    # TODO: singleton field event.function is very similar to objective in
    # OptimizationProblem. And at_time. Need to be able to define such singleton
    # assignment fields by name so there's no clash for repeated symbols. 
    update = MatchedField(ODESystem.state, direction=Direction.output)
    # make[mode_var] = SomeModeSubclass
    # actually, just update it
    #make = MatchedField(ODESystem.finite_state)
    # terminate = True -> return nan instead of update?
    def __init_subclass__(cls, dt=0., **kwargs):
        if dt:
            cls.function

# this should just provide the capabiility to overwrite make (or whatever sets control)
# and dot based on condition...
# needs to inject on creation? Or is TrajectoryAnalysis implementation expected to
# iterate Modes and inject? Then can add dot and make to copy_fields
class Mode(Model, inner_to=ODESystem,):
    """
    convenience for defining conditional behavior for state dynamics and/or controls
    depending on `condition`. No condition to over-write behavior, essentially a way to
    do inheritance for ODESystems which is otherwise hard? Can this be used instead of
    deferred subsystems? Yes but only for ODESystems..
    """
    pass
    action = MatchedField(ODESystem.modal, direction=Direction.internal)


def LTI(A, B=None, dt=0., dt_plant=False, name="LTISystem", ):

    attrs = ModelType.__prepare__(name, (ODESystem,))
    attrs["A"] = A
    state = attrs["state"]
    x = state(shape=A.shape[0])
    attrs["x"] = x
    xdot = A@x
    if dt <= 0. and dt_plant:
        raise ValueError

    if B is not None:
        attrs["B"] = B
        K = attrs["parameter"](shape=B.T.shape)
        attrs["K"] = K

        if dt and not dt_plant:
            # sampled control
            u = state(shape=B.shape[1])
            attrs["u"] = u
            #attrs["initial"][u] = -K@x
        else:
            # feedback control matching system
            u = -K@x
            attrs["output"].u = u

        xdot += B@u

    if not (dt_plant and dt):
        attrs["dot"][x] = xdot

    plant = ModelType(name, (ODESystem,), attrs)

    if dt:
        dt_attrs = InnerModelType.__prepare__("DT", (plant.Event,))
        #dt_attrs["function"] = np.sin(plant.t*np.pi/dt)
        dt_attrs["at_time"] = slice(None, None, dt)
        if dt_plant:
            from scipy.signal import cont2discrete
            if B is None:
                B = np.zeros((A.shape[0], 1))
            Ad,Bd,*_ = cont2discrete((A,B,None,None), dt=dt)
            dt_attrs["update"][dt_attrs["x"]] = (Ad - Bd@K) @ x
        elif B is not None:
            dt_attrs["update"][dt_attrs["u"]] = -K@x
            #dt_attrs["update"][dt_attrs["x"]] = x
        DTclass = InnerModelType("DT", (plant.Event,), attrs=dt_attrs, )

    return plant

