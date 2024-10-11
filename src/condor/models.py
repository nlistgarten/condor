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
from dataclasses import asdict, dataclass, field, replace
from condor._version import __version__
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
class BaseModelMetaData:
    model_name: str = ""
    independent_fields: list = field(default_factory=list)
    matched_fields: list = field(default_factory=list)
    assigned_fields: list = field(default_factory=list)

    input_fields: list = field(default_factory=list)
    output_fields: list = field(default_factory=list)
    internal_fields: list = field(default_factory=list)

    input_names: list = field(default_factory=list)
    output_names: list = field(default_factory=list)

    submodels: list = field(default_factory=list)
    embedded_models: dict = field(default_factory=dict)
    bind_embedded_models: bool = True

    inherited_items: dict = field(default_factory=dict)
    template: object = None



    # assembly/component can also get children/parent
    # assembly/components get inheritance rules? yes, submodels don't need it -- only
    # attach to primary. or should events be assemblies to re-use them? probably not --
    # re-use other things and embed but update itself is ODE system specific

    # trajectory analysis gets an "exclude_modes" and "exclude_events" list?
    # no -- the metaclass gets kwarg and copies the events that will be kept

    @property
    def all_fields(self):
        return self.input_fields + self.output_fields + self.internal_fields

    @property
    def dependent_fields(self):
        return [f for f in self.all_fields if f not in self.independent_fields]



# appears in __new__ as attrs
class BaseCondorClassDict(dict):
    def __init__(
            self, *args, model_name='',
            #copy_fields=[], primary=None,
            meta = None, **kwargs
    ):
        super().__init__(*args, **kwargs, dynamic_link = self.dynamic_link)
        self.from_outer = {}
        self.meta = meta
        #self.copy_fields = copy_fields
        #self.primary = primary
        #if primary is None:
        #    self.primary_independent_fields = []
        #else:
        #    self.primary_independent_fields = [f for f in primary._meta.independent_fields]
        self.kwargs = kwargs
        self.args = args


    def set_outer(self, **from_outer):
        self.from_outer = from_outer
        self.meta.independent_fields.extend([
            v for k, v in from_outer.items()
            if isinstance(v, IndependentField)# and k not in self.copy_fields
        ])

    def dynamic_link(self, **kwargs):
        """
        replace the string values of a dictionary with the actual object with that
        address

        dynamic_link(xx='state.x')

        returns a dictionary where the key 'xx' points to the state with name 'x'
        """

        for k, v in kwargs.items():
            path_list = v.split('.')
            use_val = self.__getitem__(path_list[0])
            for path_step in path_list[1:]:
                use_val = getattr(use_val, path_step)
            if isinstance(use_val, FreeField):
                use_val = use_val(name=k)
            kwargs[k] = use_val

        return kwargs

    def __getitem__(self, *args, **kwargs):
        return super().__getitem__(*args, **kwargs)

    def __setitem__(self, attr_name, attr_val):
        print(f"setting {attr_name} to {attr_val}")
        if isinstance(attr_val, IndependentField):
            self.meta.independent_fields.append(attr_val)
        if isinstance(attr_val, MatchedField):
            self.meta.matched_fields.append(attr_val)
        if isinstance(attr_val, AssignedField):
            self.meta.assigned_fields.append(attr_val)
        if isinstance(attr_val, Field):
            if attr_val._direction == Direction.input:
                self.meta.input_fields.append(attr_val)
            if attr_val._direction == Direction.output:
                self.meta.output_fields.append(attr_val)
            if attr_val._direction == Direction.internal:
                self.meta.internal_fields.append(attr_val)
        if isinstance(attr_val.__class__, BaseModelType):
            if attr_val.name:
                self.meta.embedded_models[attr_val.name] = attr_val
                super().__setitem__(attr_val.name, attr_val)
            else:
                self.meta.embedded_models[attr_name] = attr_val
        if isinstance(attr_val, backend.symbol_class):
            # from a IndependentField
            known_symbol_type = False

            for free_field in self.meta.independent_fields:# + self.primary_independent_fields:
                symbol = free_field.get(backend_repr=attr_val)
                # not a list (empty or len > 1)
                if isinstance(symbol, BaseSymbol):
                    known_symbol_type = True
                    #if symbol.name and symbol.name != attr_name:
                    #    raise NameError(f"Symbol on {free_field} has name {symbol.name} but assigned to {attr_name}")
                    if symbol.name:
                        pass
                    elif attr_name:
                        symbol.name = attr_name
                    else:
                        symbol.name = f"{field._model_name}_{field._name}_{field._symbols.index(symbol)}"
                    # pass attr if field is bound (_model is a constructed Model
                    # class, not None), otherwise will get added later after more
                    # processing
                    break

            # TODO: MatchedField in "free" mode
            # TODO: from the output of a subsystem? Does this case matter?

            if not known_symbol_type:
                #print("unknown symbol type", attr_name)#, attr_val)
                pass
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


        return super().__setitem__(attr_name, attr_val)


# TODO: a way to update options -- options should become attributes on implementation
# that are used on every call
class Options:
    """
    re-writing implementations to be backend agnostic (through "wrapper" layer ), the
    option class won't need to be named after a backend; maybe just take __solver__
    attribute to map the solver funciton [it could actually be the callable?
    scipy.minimize, lol sorry Kenny, you can change it after release once I'm bored]
    implementations dictionary needs to key on solver and be the arg/kwarg location for
    the numeric callbacks (for this model/paramters) and a few other things like initial
    state. Current version is pretty close, hopefully

    Goal: ensure API works to pass in *args and **kwargs easily; Do later
    probably need a __args attr and then rest as kwargs. Common pattern of reserving
    wordds in user code. Also applies to IO name check, assigned field

    to support changing solver options within an ipython session, maybe implementation
    actually has a create_solver and call_solver or something -- creating solver also
    caches any numeric callbacks

    All backend implementations must ship with reasonable defaults.

    Actually, don't need to inherit if it's backend agnostic -- can just be by name
    (which is what Django does)
    """
    pass
    # TODO: do meta programming so implementation enums are available without import?

    # TODO: allow a generic options for non-solver specific options?

#class MetaclassType(type):

#class BaseModelType(MetaclassType, ):
class BaseModelType(type):
    """
    Metaclass for Condor  model
    """
    metadata_class = BaseModelMetaData
    dict_class = BaseCondorClassDict

    # not strictly necessary here since BaseModel doesn't need to exist but convenient
    # to have the logic to automatically fill this out on all subclasses of
    # BaseModelType
    baseclass_for_inheritance = None
    def __init_subclass__(cls, **kwargs):
        cls.baseclass_for_inheritance = None
        # backreference for baseclass (e.g., ModelTemplate, Model, etc) not created until 

        # potential additional metadata:
        # these are only for templates, for 

        # class dictionary subclass to over-write __set_item__ (to handle particular
        # attribute types/fields/etc, -- may be able to use callback on metclass or even
        # field

        # similar for meta attributes, will ultimately get used by __prepare__ or
        # __new__, I think?



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
    def __prepare__(cls, name, bases, meta=None, **kwds):
        print(f"BaseModelType.__prepare__(cls={cls}, name={name}, bases={bases}, kwargs={kwds})")
        cls_dict = cls.prepare_create(name, bases, meta=meta, **kwds)
        if not cls_dict.meta.template:
            return cls_dict
        cls.prepare_populate(cls_dict)
        return cls_dict


    @classmethod
    def prepare_create(cls, name, bases, meta=None, **kwds):
        #sup_dict = super().__prepare__(cls, name, bases, **kwds)
        sup_dict = {} # equivalent to super().__prepare__ unless I break something

        if meta is None:
            meta = cls.metadata_class(
                #primary=primary
                model_name = name,
            )

        for base in bases:
            print(base)
            if isinstance(base, BaseModelType):
                meta.template = base
                break

        cls_dict = cls.dict_class(
            model_name=name,
            meta = meta,
            #copy_fields = kwds.pop('copy_fields', []),
            #primary = kwds.pop('primary', None),
            **sup_dict,
        )

        # TODO: may need to search MRO resolution, not just bases, which without mixins
        # are just singletons. For fields and submodel classes, since each generation of
        # class is getting re-inherited, this is sufficient. 

        return cls_dict

    @classmethod
    def prepare_populate(cls, cls_dict):
        meta = cls_dict.meta
        name = meta.model_name
        _dict = meta.template.__dict__
        for k, v in _dict.items():
            # inherit fields from base -- bound in __new__
            if isinstance(v, Field):
                v_class = v.__class__
                v_init_kwargs = v._init_kwargs.copy()
                for init_k, init_v  in v._init_kwargs.items():
                    if isinstance(init_v, Field):
                        # TODO: is it OK to always assume cls_dict has the proper reference injected
                        v_init_kwargs[init_k] = cls_dict[init_v._name]
                cls_dict[k] = v.inherit(
                    name, field_type_name=k, **v_init_kwargs
                )
                cls_dict[k].prepare(cls_dict)

            # inherit submodels model from base, create reference now, bind in new
            elif isinstance(v, BaseModelType):
                if v.primary is base and v in base._meta.submodels:
                    # submodel inheritance & binding in new
                    cls_dict[k] = v
            elif isinstance(v, BaseSymbol):
                # should only be used for placeholder and placeholder-adjacent
                print(f"Element not inheriting {k}={v} from {meta.template} to {name}")
                cls_dict[k] = v
            elif isinstance(v, backend.symbol_class):
                # only used for time?
                print(f"Symbol inheriting {k}={v} from {meta.template} to {name}")
                cls_dict[k] = v

            if k in cls_dict:
                meta.inherited_items[k] = cls_dict[k]



        return cls_dict

    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def __repr__(cls):
        if cls.__bases__:
            return f"<{cls.__bases__[0].__name__}: {cls.__name__}>"
        else:
            return f"<{cls.__name__}>"

    @classmethod
    def __pre_super_new__(
        cls, name, bases, attrs,
        **kwargs
    ):
        """return new name, bases, attrs to be passed to type.__new__
        and post_kwargs
        """
        # perform as much processing as possible before caller super().__new__
        # by building up super_attrs from attrs; some operations need to operate on the
        # constructed class, like binding fields and submodels

        super_attrs = {}

        backend_options = {}
        matched_fields = []

        # used to find free symbols and replace with symbol, seeded with outer model


        # will be used to pack arguments in (inputs) and unpack results (output), other
        # convenience
        # TODO: better reserve word check? see check attr name below

        if name == "Coupling":
            #breakpoint()
            pass


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

            if callable(attr_val) and attr_val == attrs.dynamic_link:
                # don't pass dynamic_link -- don't think we want this but maybe we do
                continue


            if isinstance(attr_val, backend.symbol_class):
                # from a IndependentField
                known_symbol_type = False
                for free_field in attrs.meta.independent_fields:
                    symbol = free_field.get(backend_repr=attr_val)
                    # not a list (empty or len > 1)
                    if isinstance(symbol, BaseSymbol):
                        known_symbol_type = True
                        if attr_name not in bases[-1].__dict__:
                            attr_val = symbol
                        # pass attr if field is bound (_model is a constructed Model
                        # class, not None), otherwise will get added later after more
                        # processing
                        pass_attr = free_field._model is not None
                        if pass_attr:
                            # TODO is this kind of defensive checking useful?
                            assert issubclass(free_field._model, primary)
                        break
            if isinstance(attr_val, BaseModelType):
                # not sure if this should really be Base or just Model? Or maybe
                # submodel by now...  
                # TODO check isinstance type
                if attr_val.primary in bases and attr_val in attr_val.primary._meta.submodels:
                    # handle submodel classes below, don't add to super
                    continue

            # Options, Subodel, and IndependentSymbol are not added here
            if pass_attr:
                check_attr_name(attr_name, attr_val, super_attrs, bases)
                super_attrs[attr_name] = attr_val

        # before calling super new, process fields

        for field in attrs.meta.independent_fields:
            for symbol_idx, symbol in enumerate(field):
                # add names to symbols -- must be an unnamed symbol without a reference
                # assignment in the class
                if not symbol.name:
                    symbol.name = f"{field._model_name}_{field._name}_{symbol_idx}"

        for matched_field in attrs.meta.matched_fields:
            for matched_symbol in matched_field:
                matched_symbol.update_name()

        # symbols from input and input fields are added directly to model
        # previously, all fields were  "finalized" by creating dataclass
        # ODESystem has no implementation and not all the fields should be finalized 
        # Events may create new parameters that are canonical to the ODESystem model,
        # and should be not be finalized until a trajectory analysis, which has its own
        # copy. state must be fully defined by ODEsystem and can/should be finalized.
        # What about output? Should fields have an optional skip_finalize or something
        # that for when an submodel class may modify it so don't finalize? And is that the
        # only time it would come up?
        # TODO: review this 

        for input_field in attrs.meta.input_fields:
            for in_symbol in input_field:
                in_name = in_symbol.name
                check_attr_name(in_name, in_symbol, super_attrs, bases)
                super_attrs[in_name] = in_symbol
                attrs.meta.input_names.append(in_name)

        for internal_field in attrs.meta.internal_fields:
            internal_field.create_dataclass()

        for output_field in attrs.meta.output_fields:
            for out_symbol in output_field:
                out_name = out_symbol.name
                check_attr_name(out_name, out_symbol, super_attrs, bases)
                super_attrs[out_name] = out_symbol
                attrs.meta.output_names.append(out_name)
            output_field.create_dataclass()

        # process docstring / add simple equation
        lhs_doc = ', '.join([out_name for out_name in attrs.meta.output_names])
        arg_doc = ', '.join([arg_name for arg_name in attrs.meta.input_names])
        orig_doc = attrs.get("__doc__", "")
        super_attrs["__doc__"] = "\n".join([orig_doc, f"    {lhs_doc} = {name}({arg_doc})"])

        post_kwargs = dict(
            backend_options = backend_options,
        )
        return name, bases, super_attrs, post_kwargs


    def __new__(
        cls, name, bases, attrs, 
        **kwargs
    ):
        print(f'  BaseModelType.__new__(mcs={cls}, name={name}, bases={bases}, attrs=[{attrs}], {kwargs})' )
        # case 1: class Model -- provides machinery to make subsequent cases easier to
        # implement.
        # case 2: ____ - library code that defines fields, etc that user code inherits
        # maybe call this model template?
        # from (case 3+). Implementations are tied to 2? "Library Models"?
        # case 3: User Model - inherits from ____, defines the actual model that is
        # being analyzed
        # case 4: Subclass of user model to extend it?
        # I don't think Submodel deviates from this, except perhaps disallowing
        # case 4
        # Generally, bases arg will be len <= 1. Just from the previous level. Is there
        # a case for mixins? Submodel approach seems decent, could repeat for
        # WithDeferredSubsystems, then Model layer is quite complete 

        # TODO: add support for inheriting other models -- subclasses are
        # modifying/adding (no deletion? need mixins to pre-build pieces?) to parent
        # classes.  case 4
        # fields would need to combine _symbols


        creating_base_class_for_inheritance = (
            cls.__name__.replace(name, "") == "Type"
            and cls.baseclass_for_inheritance is None
        )
        if creating_base_class_for_inheritance:
            print("creating base class for inheritance")
            print(f"cls.baseclass_for_inheritance={cls.baseclass_for_inheritance}")


        super_name, super_bases, super_attrs, post_kwargs = cls.__pre_super_new__(
            name, bases, attrs,
            **kwargs
        )



        new_cls = super().__new__(
            cls, super_name, super_bases, super_attrs,
            **kwargs,
        )
        new_cls._meta = replace(attrs.meta)


        cls.__post_super_new__(
            new_cls, bases, attrs,
            **post_kwargs,
        )
        # TODO: I'm surprised that this is necessary to avoid duplciating the mutable
        # default subclasses list; but this works on the cases I have

        if name == "Sellar":
            #breakpoint()
            print("Sellar class ID at BaseModelType", id(new_cls))
            print("Sellar._meta class ID at BaseModelType", id(new_cls._meta))
            print("Sellar._meta BaseModelType", new_cls._meta)
            pass

        if creating_base_class_for_inheritance:
            print("creating base class for inheritance")
            cls.baseclass_for_inheritance=new_cls

        return new_cls

    @classmethod
    def __post_super_new__(
        cls, new_cls, bases, attrs,
        backend_options = {},
        **kwargs
    ):
        """
        mutate new_cls as needed for finalization
        """

        for attr_name, attr_val in attrs.items():
            if isinstance(attr_val, Field):
                attr_val.bind(attr_name, new_cls)

        # creating an SubmodelClass
        # primary kwarg is added to Submodel  template definition or for subclasses
        # of submodel templates, in Submodel.__new__. For subclasses of template,
        # reference to base is inner_through


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


        if implementation is None and new_cls._meta.template:
            implementation = getattr(
                backend.implementations,
                new_cls._meta.template.__name__,
                None
            )

        if implementation is not None:
            cls.finalize_input_fields(new_cls)
            new_cls.implementation = implementation(new_cls, **backend_option)

            if new_cls.__name__ == "Sellar":
                #breakpoint()
                pass

            for field in attrs.meta.all_fields:
                field.bind_dataclass()

        for field in attrs.meta.independent_fields:
            for symbol_idx, symbol in enumerate(field):
                setattr(new_cls, symbol.name, symbol)


    def finalize_input_fields(cls):
        for input_field in cls._meta.input_fields:
            input_field.create_dataclass()


    def register(cls, subclass):
        cls._meta.subclasses.append(subclass)


def check_attr_name(attr_name, attr_val, super_attrs, bases):
    # TODO: this needs to also be called to check on bases dict to prevent creating a
    # symbol with the name of state? I think?

    if attr_name in ['__module__', '__qualname__', '__doc__']:
        return

    if len(bases) == 1 and bases[0] == BaseModel:
        # Skip model types
        # TODO: is there a better way to capture it's a model type? No symbols, but
        # fields?
        return


    attr_val_backend = getattr(attr_val, 'backend_repr', None)
    clash_value = super_attrs.get(attr_name, None)

    if (
        (
            isinstance(attr_val_backend, backend.symbol_class)
            and isinstance(clash_value, backend.symbol_class)
        )
    ):
        if backend.utils.symbol_is(attr_val_backend, clash_value):
            return

    elif attr_val_backend == clash_value:
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
                or isinstance(attr_val, backend.symbol_class)
            )
        ):
            in_bases = True
            break

    if (attr_name in super_attrs and super_attrs[attr_name] is not attr_val) or in_bases:
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


class BaseModel(metaclass=BaseModelType):
    pass



class ModelTemplateType(BaseModelType):
    """Define a Model Template  by subclassing Model, creating field types, and writing an
    implementation."""

    user_model_metaclass = None
    user_model_baseclass = None
    def __init_subclass__(cls, **kwargs):
        cls.user_model_metaclass
        cls.user_model_baseclass
        super().__init_subclass__(**kwargs)

    @classmethod
    def is_user_model(cls, bases):
        return (
            cls.baseclass_for_inheritance is not None and
            cls.baseclass_for_inheritance not in bases
        )


    @classmethod
    def __prepare__(
        cls, name, bases,
        **kwargs
    ):
        if cls.baseclass_for_inheritance and cls.baseclass_for_inheritance not in bases:
            print("dispatch __prepare__ for user model")
            # user model
            return cls.user_model_metaclass.__prepare__(
                name, bases + (cls.user_model_baseclass,),
                **kwargs
            )
        else:
            return super().__prepare__(name, bases, **kwargs)

    @classmethod
    def __pre_super_new__(
        cls, name, bases, attrs,
        **kwargs
    ):
        ret_name, ret_bases, ret_attrs, ret_kwargs,  = super().__pre_super_new__(
            name, bases, attrs
        )

        super_attrs = {}
        for attr_name, attr_val in ret_attrs.items():
            if attr_name not in attrs.meta.inherited_items:
                print("template not passing on", attr_name, "=", attr_val)
                super_attrs[attr_name] = attr_val

        return name, ret_bases, super_attrs, ret_kwargs

    def __new__(cls, name, bases, attrs, **kwargs):

        if cls.is_user_model(bases):
            print("dispatch __new__ for user model", name)
            # user model
            return cls.user_model_metaclass(
                name, bases + (cls.user_model_baseclass,), attrs, 
                **kwargs
            )

        new_cls = super().__new__(cls, name, bases, attrs, **kwargs)


        return new_cls


        if model_name == "ModelTemplate":
            # case 1, essentially placeholder to enable next case
            # actually base model type should do less, and depth-specific __new__ and
            # __prepare__ logic
            return {} # except new is still going through BaseModelType??
        # need to figure out good dispatch mechanism; maybe BaseModelType can identify
        # reserved model_names or even , ModelTemplate and M then just call cls.__
        # this might work? cls.__name__.replace(model_name, "")  == "Type"
            pass
        if ModelTemplate in bases: # and not some flag
            # case 2, library defined template 
            # presumably this metaclass is where this should be;
            pass
        else:
            # case 3, different metaclass and inject Model into bases too? 
            pass
    pass

class ModelTemplate(BaseModel, metaclass=ModelTemplateType):
    placeholder = FreeField(Direction.internal)
    pass



class ModelType(BaseModelType):
    """Type class for user models"""

    def __repr__(cls):
        if cls.__bases__:
            return f"<{cls._meta.template.__name__}: {cls.__name__}>"

    @classmethod
    def __prepare__(
        cls, model_name, bases,
        bind_embedded_models=True, name="", 
        **kwargs
    ):
        if name:
            model_name = name
        meta = cls.metadata_class(
            model_name=model_name,
            bind_embedded_models=bind_embedded_models
        )
        print(f"ModelType.__prepare__(model_name={model_name}, bases={bases}, **{kwargs}), meta={meta}")

        return super().__prepare__(model_name, bases, meta=meta, **kwargs)


    @classmethod
    def __pre_super_new__(
        cls, model_name, bases, attrs,
        bind_embedded_models=True, name="", 
        **kwargs
    ):
        if not name:
            name = model_name

        ret_name, ret_bases, ret_attrs, ret_kwargs,  = super().__pre_super_new__(
            name, bases, attrs
        )

        return name, ret_bases, ret_attrs, ret_kwargs

    @classmethod
    def is_user_model(cls, bases):
        return (
            cls.baseclass_for_inheritance is not None and
            cls.baseclass_for_inheritance in bases
        )


    def __new__(
        cls, name, bases, attrs,
        #bind_embedded_models=True, name="", 
        **kwargs
    ):
        my_super = super()
        new_cls =  super().__new__(cls, name, bases[1:], attrs, **kwargs)

        if name == "Sellar":
            print("Sellar class ID at ModelType", id(new_cls))
            print("Sellar._meta class ID at ModelType", id(new_cls._meta))
            print("Sellar._meta ModelType", new_cls._meta)
            #breakpoint()
        if cls.is_user_model(bases):
            for submodel in new_cls._meta.template._meta.submodels:
                # TODO inherit submodels
                pass

        return new_cls

    @classmethod
    def __post_super_new__(
        cls, new_cls, bases, attrs,
        **kwargs
    ):
        super().__post_super_new__(new_cls, bases, attrs, **kwargs)

class Model(BaseModel, metaclass=ModelType):
    """handles binding etc. for user models"""
    def __init__(self, *args, name='', **kwargs):
        cls = self.__class__
        self.name = name
        # bind *args and **kwargs to to appropriate signature
        # TODO: is there a better way to do this?
        input_kwargs = {}
        for input_name, input_val in zip(cls._meta.input_names, args):
            if input_name in kwargs:
                raise ValueError(f"Argument {input_name} has value {input_val} from args and {kwargs[input_name]} from kwargs")
            input_kwargs[input_name] = input_val
        input_kwargs.update(kwargs)
        missing_args = []
        extra_args = []
        self.input_kwargs = {}
        for name in cls._meta.input_names:
            if name in input_kwargs:
                self.input_kwargs[name] = input_kwargs[name]
            else:
                missing_args.append(name)


        # TODO: skip this one with a flag?
        for key in kwargs:
            if key not in self.input_kwargs:
                extra_args.append(key)

        if missing_args or extra_args:
            error_message = f"While calling {cls.__name__}, "
            if extra_args:
                error_message += f"recieved extra arguments: {extra_args}"
            if extra_args and missing_args:
                error_message += " and "
            if missing_args:
                error_message += f"missing arguments: {missing_args}"
            raise ValueError(error_message)

        # TODO: check bounds on model inputs?

        # pack into dot-able storage, over-writting fields and symbols
        self.bind_input_fields()

        cls.implementation(self, *list(self.input_kwargs.values()))

        # generally implementations are responsible for binding computed values.
        # implementations know about models, models don't know about implementations
        if False:
            self.output_kwargs = output_kwargs = {
                out_name: getattr(self, out_name)
                for field in cls._meta.output_fields
                for out_name in field.list_of('name')
            }

        self.bind_embedded_models()


    def bind_input_fields(self):
        cls = self.__class__
        all_values = list(self.input_kwargs.values())

        slice_start = 0
        for field in cls._meta.input_fields:
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
        for output_name in cls._meta.output_names:
            yield getattr(self, output_name)

    def __repr__(self):
        return f"<{self.__class__.__name__}: " + ", ".join([f"{k}={v}" for k, v in self.input_kwargs.items()]) + ">"

    def bind_embedded_models(model_instance):
        # TODO: how to have models cache previous results so this is always free?
        # Can imagine a parent model with multiple instances of the exact same
        # sub-model called with different parameters. Would need to memoize at least
        # that many calls, possibly more.

        # if model instance created with `name` attribute, result will be bound to that
        # name not the assigned name, but assigned name can still be used during model
        # definition
        model = model_instance.__class__
        model_assignments = {}

        fields = [
            field
            for field in model._meta.input_fields + model._meta.output_fields
            if isinstance(field, IndependentField)
        ]
        for field in fields:
            model_instance_field = getattr(model_instance, field._name)
            model_instance_field_dict = asdict(model_instance_field)
            model_assignments.update({
                elem.backend_repr: val
                for elem, val in zip(field, model_instance_field_dict.values())
            })

        if not model._meta.bind_embedded_models:
            return

        for embedded_model_ref_name, embedded_model_instance in model._meta.embedded_models.items():
            embedded_model = embedded_model_instance.__class__
            embedded_model_kwargs = {}

            for field in embedded_model._meta.input_fields:
                bound_field = getattr(embedded_model_instance, field._name)
                bound_field_dict = asdict(bound_field)
                for k,v in bound_field_dict.items():
                    if not isinstance(v, backend.symbol_class):
                        embedded_model_kwargs[k] = v
                    else:
                        value_found = False
                        for kk, vv in model_assignments.items():
                            if backend.utils.symbol_is(v, kk):
                                value_found = True
                                break
                        embedded_model_kwargs[k] = vv
                        if not value_found:
                            embedded_model_kwargs[k] = backend.utils.evalf(
                                v, model_assignments
                            )

            bound_embedded_model = embedded_model(**embedded_model_kwargs)
            setattr(model_instance, embedded_model_ref_name, bound_embedded_model)


ModelTemplateType.user_model_metaclass = ModelType
ModelTemplateType.user_model_baseclass = Model

"""
Do we need class inheritance?

Case 1: defining a new model template -- very fiew fields, just copy and paste;
manipulate and re-use implementaitons etc

Case 2: defining a new user model -- use sub-models or functions that perform
declarative operations for you? Need to think about this, I guess it should be possible
to support but probably better performance for functions. Could only be the same type??
I think relies on a get_or_create so variables can be re-used. Or maybe it really needs
substitution mechanism from placeholders which could be fine, this pattern came from
inheriting fields...

class inheritance would be a mechanism to translate ODE to DAE -- seems useful.

Case 3: Submodel version of same...
Not sure if it's a true inner model, but definitely need some capability to re-use both
Model and ModelTemplate on different "outer" model/template. 

rename:
submodel --> embedded_model, generally no back reference
innermodel --> submodel, so inner_to is supermodel? parent? need to distinguish from
assembly relationship, I think? although I did wonder if assembly is just a special type
of submodel -- in both cases, want to generalize so templates can handle multiple
sub-types? Thinking of LinCov and adding noise to event, but maybe I only want to
support ODE vs DAE which is slightly different?

new gascon's library of components will need something like condor-flight's
"placeholder" field. Is this a field that is always useful for building library's? and
the metatype that goes through the output fields and does the substitutions when users
create their version? -- this is another *type* of declaration that needs a name. It's
more than a model template, but not yet a user model.

and, is it always used in assembly models? condor flight is a very structured assembly
-- planet (or really, baricenter/ central body), vehicles that must have planet parent,
then event/mode on vehicles... seems like there may be a relationship btwn placeholder
and assembly.

also, simulate "freezes" the assembly, which has been extremely useful for free drift
stuff, may be a nice mechanism for gasp missions, although maybe the "segment assembly"
and "operator" approach could be even better?? 

I guess we can just expect to provide class methods on assembly models that perform
symbolic operations on what currently exists? I think at that point, placeholders would
have been substituted for their values or appropriate field elements; so AssemblyModel
type can provide utility methods for tree traversal but users just define their own
thing. Maybe it needs to get wrapped in something like a model? not so different from
Vehicle's solve -

Or an "Airplane" submodel that acts on root component (maybe an OML body, ? maybe user
choice) which freezes thing, then Airplane has to have inputs bound and has the instance
methods for operating? But actually something like aero and propulsion still need to
create something like an explicit system (maybe it would be unique for gascon, something
like "dynamic/performance component" or something). 

Maybe "Design" off of the root of the node? And that's what gets named by a particular
airplane? And root is just body. And then can even tree off the design for the
performance? Point performance modules, a "mission" is a path on the tree from design,
ground roll etc. can branch from mission to do OEI, etc.


Re: simulate to freeze:
Is it more efficient to have each free drift go through the whole sim, instead of saving
the state and re-initializing each segment separately? For RPOD trajectory design,
perhaps that would be a time when forward mode would be better... and is
forward-over-adjoint ALWAYS better? or is there a scenario in which forward-over-forward
(directional) is better?


both InnerModel and Model have the same 3 cases: core, template, user; Model "knows"
about InnerModel -- can we fix that?
Is there a better way to handle the 3 cases, or is it acceptable that the same metaclass
has to do all 3?

There are possibly 2 more: Model with placeholder (LibraryModel??), and AssemblyModel.
It sould be nice if all 3 sub-types could be mixed-and-matched.

Also, want a singleton placeholder element or descriptor -- basically a placeholder
field on models where users aren't adding elements. could be a detached field in Condor
that template makers can use? Actually, the use cases I'm thinking of
(OptimizationProblem.objective, Gascon.OMLComponent.Aero.{CL, CD},
Gascon.Propulsor.Propulsion.thrust, ) are outputs with
defaults -- its own field.

what about tf (I guess could be an expression)

submodels from atmosphere etc models from Condor-Flight?


placeholders: elements defined by library creators to allow user inputs; condor provides
substitution mechanisms, etc. ~ expected uer input

submodels are mdoels that don't make sense w/o their parent, maybe add configuration (like
singleton). inner_to arg becomes modifying? because submodels only exist to modify their
parent?  maybe submodel modifies its superior? 


A user Model with a parent, a ModelTemplate with a parent is a submodel
 (possibly only Assembly models will allow user model to set a parent)

assemblies make sense on their own, so user must define relationship, and be able to
attach/be adopted/assign/etc. after model definition. -- hook for symbolic processing at
attachment?
are assmeblies just a special type for library makers to define useful datastructures?
need a submodel/type to actually operate on them and do computation...not sure what an
assembly of (mix and match) explicit systems, algebraic, etc would even be!
Actually, if computation is always on some end-point submodel, can do all processing
htere? Yes, assembly components are JUST a datastructure ot define inputs, it solves the
the sensor model problem from LinCov demo to automatically handle creating parameters, a
few things about accessing namespace, etc. 

flags/controls to conntrol inheritance:
"as_template" flag -- dry way to create new but related templates similar to user model
inheritance
parent relationship inheritance: what gets shared vs copied vs. ??
copied --> computation/end-point submodel

TrajectoryAnalysis should get extra flags for keep/skip events and modes




"""

@dataclass
class SubmodelMetaData(BaseModelMetaData):
    # primary only needed for submodels, maybe also subclasses?
    primary: object = None
    copy_fields: list = field(default_factory=list)
    subclasses: list = field(default_factory=dict)


class SubmodelTemplateType(
    ModelTemplateType
):
    metadata_class = SubmodelMetaData

    @classmethod
    def __prepare__(
        cls, model_name, bases,
        primary, copy_fields=None,
        **kwds
    ):
        if copy_fields is None:
            copy_fields = []
        meta = cls.metadata_class(
            model_name=model_name,
            primary=primary, copy_fields=copy_fields,
        )
        cls_dict = super().__prepare__(model_name, bases,  meta=meta, **kwds)
        return cls_dict
        _dict = meta.template.__dict__
        for k, v in _dict.items():
            pass

    def __new__(
        cls, name, bases, attrs,
        primary, copy_fields = None,
        **kwargs
    ):
        new_cls =  super().__new__(cls, name, bases[:], attrs, **kwargs)
        if cls.baseclass_for_inheritance is not None and cls.baseclass_for_inheritance is not new_cls:
            new_cls._meta.primary._meta.submodels.append(new_cls)
        return new_cls

class SubmodelTemplate(ModelTemplate, metaclass=SubmodelTemplateType, primary=None):
    pass



class OldSubmodelType:
    def __iter__(cls):
        for subclass in cls.subclasses:
            yield subclass

    @classmethod
    def __prepare__(cls, model_name, bases, primary=None, original_class=None, **kwds):
        case1 = model_name == "Submodel"
        case2 = primary is not None and original_class is not None
        case3 = not case1 and not case2

        if case1 or case2:
            return super().__prepare__(
                model_name, bases, primary=primary, **kwds
            )

        if case3:
            return bases[0].__prepare__(
                model_name, bases, primary=primary, **kwds
            )


            # if base is an submodel, make reference  of outer model attributes
            # and copy fields from class __copy_fields__ kwarg definition. This allows
            # submodel field to modify local copy without affecting outer model
            if base is not BaseModel and base.primary:
                for attr_name, attr_val in base.primary.__original_attrs__.items():
                    # TODO: document copy fields? can this get DRY'd up?
                    # don't like that symbol copying is here, maybe should be method on
                    # field?
                    if attr_name in base.__copy_fields__:
                        field_class = attr_val.__class__
                        v_init_kwargs = attr_val._init_kwargs.copy()
                        for init_k, init_v  in attr_val._init_kwargs.items():
                            if isinstance(init_v, Field):
                                # TODO not sure this will be the best way to remap connected
                                # fields based on inheritance, but maybe sufficient?
                                if not init_v._name in base.__copy_fields__ and _base.primary and init_v._name in base.primary.__dict__:
                                    get_from = base.primary.__dict__
                                else:
                                    get_from = cls_dict
                                v_init_kwargs[init_k] = get_from[init_v._name]
                        # I know it has no field references
                        copied_field = attr_val.inherit(
                            name, field_type_name=attr_name, **v_init_kwargs
                        )
                        copied_field._symbols = [sym for sym in attr_val]
                        copied_field._count += sum(copied_field.list_of('size'))
                        cls_dict[attr_name] = copied_field
                        continue

                    cls_dict[attr_name] = attr_val
                # TODO: document __from_outer__?
                cls_dict.set_outer(**base.primary.__original_attrs__)
        # TODO: is it better to do submodel magic in SubmodelType.__prepare__?


    @classmethod
    def __pre_super_new__(
        cls, model_name, bases, attrs, bind_embedded_models=True, name="", 
        primary=None, inner_through = None, copy_fields = None,
        **kwargs
    ):
        return super().__pre_super_new__(cls, model_name, bases, attrs)
        # __from_outer__ attribute is attached to Submodel`s during __prepare__ to
        # give references to IndependentVariable backend_repr`s conveniently for
        # constructing Submodel symbols. They get cleaned up since they live in the
        # outer model
        #attrs_from_outer = attrs.pop('__from_outer__', {})
        attrs_from_outer = attrs.from_outer #getattr(attrs, 'from_outer', {})
        embedded_models = attrs.meta.embedded_models
        attrs.meta.bind_embedded_models = bind_embedded_models
        for attr_name, attr_val in attrs_from_outer.items():
            if attrs[attr_name] is attr_val:
                attrs.pop(attr_name)
                if embedded_models.get(attr_name, None) is attr_val:
                    embedded_models.pop(attr_name)
            else:
                # TODO: make this a warning/error? add test (really for all
                # warning/error raise)
                # Or allow silent passage of redone variables? eg copying
                # but copying is a special case that can be guarded...

                # TODO: if we put copy fields in meta can we trace through and not warn
                # on this case?
                pass
                # print(f"{attr_name} was defined on outer of {name}")



        if bases:
            super_attrs['_parent_name'] = bases[0].__name__
            if 'primary' not in kwargs and bases[0] is not BaseModel and bases[0].primary:
                kwargs['primary'] = bases[0].primary
        else:
            super_attrs['_parent_name'] = ''

        primary=kwargs.pop('primary', None)
        inner_through=kwargs.pop('inner_through', None)
        copy_fields=kwargs.pop('copy_fields', [])

        pass

    @classmethod
    def __post_super_new__(
        cls, model_name, bases, attrs, bind_embedded_models=True, name="", 
        primary=None, inner_through = None, copy_fields = None,
        **kwargs
    ):
        new_cls.primary = primary
        if primary:
            # TODO: other validation?
            if new_cls.__name__ in primary.__dict__:
                raise ValueError

            user_submodel = False
            for base in new_cls.__bases__:
                if base is not BaseModel:
                    if base in primary._meta.submodels:
                        user_submodel = True
                        break
                        # TODO: base is ~ the field that cls should be added to...
                        # This could all be in BaseModelType.__new__, primary is in kwargs

            if user_submodel:
                # register as symbol of field, not directly added
                if not inner_through:
                    raise ValueError
            else:
                # create as field
                primary._meta.submodels.append(new_cls)
                setattr(primary, new_cls.__name__, new_cls)
                new_cls.__copy_fields__ = copy_fields

        if inner_through:
            # create links
            inner_through.register(new_cls)
            new_cls.inner_through = inner_through
            new_cls.primary = inner_through.primary

        elif bases and bases[0] is not BaseModel:
            bases[0].register(new_cls)


        # Bind Fields and Submodels -- require reference to constructed BaseModel
        for attr_name, attr_val in attrs.items():

            if isinstance(attr_val, BaseModelType):
                # TODO check that this check is right...
                if attr_val.primary in bases and attr_val in attr_val.primary._meta.submodels:
                    # attr_val is an submodel to base, so this is a field-like submodel
                    # model that the user will sub-class

                    # new_cls is a user model that inherits an submodel type

                    use_bases = (Submodel,)
                    if ModelTemplate in attr_val.__bases__ and len(attr_val.__bases__) == 1:
                        pass
                    else:
                        raise NotImplemented
                        # does this work?
                        use_bases = use_bases + attr_val.__bases__

                    # TODO: can we dry up all these extra args? or is it necessary to
                    # specify these? or at leat document that this is where new features
                    # on sub models need to be added, in addition to poping from
                    # kwargs, etc. When reverting sub model stuff to Submodel's init
                    # subclass, may be more obvious

                    attr_val = SubmodelType(
                        attr_name,
                        use_bases,
                        attr_val.__original_attrs__,
                        primary = new_cls,
                        original_class = attr_val,
                        copy_fields = attr_val.__copy_fields__,
                    )
                    setattr(new_cls, attr_name, attr_val)


    def __new__(cls, model_name, bases, attrs, primary=None, original_class = None,  **kwargs):
        # case 1: Submodel definition
        # case 2: library submodel inherited to user model through user model's __new__
        # (submodel template)
        # case 3: subclass of inherited submodel -- user defined model
        case1 = model_name == "Submodel"
        case2 = primary is not None and original_class is not None
        case3 = not case1 and not case2

        if case1 or case2:
            new_cls = super().__new__(cls, model_name, bases, attrs, primary=primary, **kwargs)

        if case2:
            new_cls.original_class = original_class
            # reference to original class so uer sub-classes inherit directly, see below

        if case3:
            if len(bases) > 1:
                raise ValueError
            # going to leave inner_through and hope after clean up it wont be necessary
            # -- all metas should include template
            inner_through = bases[0]
            original_class = inner_through.original_class
            # subclass original class, inheriting
            # will inherit primary from original_class
            new_cls = original_class.__class__(model_name, (original_class,), attrs,
                                               inner_through=inner_through, **kwargs)

        return new_cls


class OldSubmodel(
    #Model, metaclass=SubmodelType, primary=None
):
    """
    Create an submodel template  by assigning an primary keyward arguement during
    model template definition. The argument references another model template (which
    becomes the "outer model"). Then a user outer model (a subclass of the submodel
    template which is bound to outer model) can create multiple subclass of the 
    Submodel by sub-classing from <outer_model>.<submodel template>
    """
    pass
