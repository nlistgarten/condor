
# TODO: figure out python version minimum

from dataclasses import dataclass, make_dataclass, fields, asdict as dataclass_asdict
import numpy as np
from enum import Enum
from condor.backends.default import backend
from condor.backends import BackendSymbolData
import importlib
import sys

# TODO **kwarg expansion for a field (with a filter?)
# TODO copy all parameters (with a filter?) from one model
# --> together, a very convenient way to connect a system with many variables
# use a dict and/or dummy system to define an "interface" which can be used to filter
# OR: pass a system itself? works for settings.conf, not sure if it's very convenient
# to implement generally (same as abandoned deferred subsystem idea)

# TODO could intercept models created within the class and create an API for the pattern
# of creating input field elements to a super-system based on input field elements that
# the sub-system require that aren't matching outputs of previous sub-systems (exact
# name matching or define dictionary where key is current subsystem input name and value
# is previous subsytem output name to connect) and 
# want to do something like 
"""
class AnyTopeofModel(co....):
    some specific setup
    
    subsystem_ref_1 = create_subsystem(SomeNormalModel, **aliases)
    # adds attributes for holding all input, and then useful subsets. ref has outputs

"""
# so this is a field, and it has to know about its parent class -- or we make parameter
# the only input field for all models and output as a required output field with others
# specified. output field maintains order of other fields elements ast hey are defined
# -- maybe output is either the built-in or a special field that combines other fields?
# so implicit and explicit still exist for algebraic but they are really accessed
# through the combo field.

# this is basically deferred model?
# aliases allow complete & explicit control of what connects to subsystem


# don't need to include API for specifying which subsystems connect to which -- just
# create a new model for each open chain, can only create a closed-loop by wrapping with
# algebraic system or optimization problem & alias to variables/implicit_output
# I guess could also have API to terminate outputs so they don't get promoted. Is this
# just literally promotes=*?


# TODO defensive chek to make sure trajectory analysis doesn't over write odemodel
# dynamic output? 
# todo inner_to is only for ode system. should backref have a better name than
# `inner_to`? Maybe even ODESystem....

# keyword expand modals? 


def asdict(obj):
    return dict(
        (field.name, getattr(obj, field.name)) for field in fields(obj) if field.init
    )

class Direction(Enum):
    """
    Used to indicate the direction of a Symbol relative to a model
    MatchedField may need to become MatchedSymbol and also use direction -- might
    be useful for DAE models, etc.
    """
    output = -1
    internal = 0
    input = 1

class FieldValues:
    """
    Base class for Field dataclasses -- may need to use issubclass/isinstance
    """
    pass

class Field:
    """
    base field

    model templates have field instances added to them which are parents of symbols used
    in models

    how to create a new Field type:
    - subclass Symbol as needed. Will automatically try to find NameSymbol for NameField,
    or provide as symbol_class kwarg in subclass definition
    - update _init_kwargs to define kwargs that should get passed during inheritance
      from Model templates to user models
    - until a better API is designed & implemented, instance attributes created during
      __init__ should start with a _ so setattr on assignedfields can filter it
    - use create_symbol to create symbols, passing **kwargs to symbol dataclass as much
      as possible to keep DRY

    """

    def _set_resolve_name(self):
        self._resolve_name = ".".join([self._model_name, self._name])

    def __init_subclass__(
            cls, symbol_class=None, default_direction=Direction.internal, **kwargs
    ):
        # TODO: make sure python version requirement is correct
        super().__init_subclass__(**kwargs)
        if symbol_class is None:
            # TODO: ensure this works for different file organizations? e.g., will it
            # find a symbol defined in another file that the field subclass has access
            # to?
            symbol_class = globals().get(cls.__name__.replace('Field', 'Symbol'))
        cls.symbol_class = symbol_class

        cls.default_direction = default_direction

    def __init__(
        self, direction=None, name='', model=None, inherit_from=None
    ):
        # TODO: currently, AssignedField types are defined using setattr, which needs
        # to know what already exists. The attributes here, like name, model, count,
        # etc, don't exist until instantiated. Currently pre-fix with `_` to mark as
        # not-an-assignment, but I guess I could just use symbol_class on value instead
        # of checking name? Anyway, really don't like needing to prefix all Field
        # attributes with _ because of AssignedField... 


        self._cls_dict = None
        self._name = name
        self._model = model
        if direction is None:
            direction = self.__class__.default_direction
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

    def bind_dataclass(self):
        mod_name = self._model.__module__
        self._dataclass.__module__ = mod_name
        if mod_name not in sys.modules:
            mod_obj = importlib.import_module(mod_name)
        else:
            mod_obj = sys.modules[mod_name]
        setattr(mod_obj, self._dataclass.__name__, self._dataclass)

    def prepare(self, cls_dict):
        self._cls_dict = cls_dict

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

    def get(self, *args, **kwargs):
        """
        return list of field symbols where every field matches kwargs
        if only one, return symbol without list wrapper
        """
        # TODO: what's the lightest-weight way to be able query? these should get called
        # very few times, so hopefully don't need to stress too much about
        # implementation
        # would be nice to be able to follow references, etc. can match be used?
        if len(args) == 1 and not kwargs:
            field_value = args[0]
            if isinstance(field_value, backend.symbol_class):
                kwargs.update(backend_repr=field_value)

        items = []
        for item in self._symbols:
            this_item = True
            for field_name, field_value in kwargs.items():
                item_value = getattr(item, field_name)
                if isinstance(item_value, backend.symbol_class):
                    if not isinstance(field_value, backend.symbol_class):
                        this_item = False
                        break
                    this_item = this_item and backend.utils.symbol_is(
                        item_value, field_value
                    )
                elif isinstance(item_value, BaseSymbol):
                    if item_value.__class__ is not field_value.__class__:
                        this_item = False
                        break
                    this_item = this_item and item_value is field_value
                else:
                    this_item = this_item and item_value == field_value
                if not this_item:
                    break
            if this_item:
                items.append(item)
        if len(items) == 1:
            return items[0]
        return items

    def flat_index(self, of_symbol):
        idx = 0
        for symbol in self:
            if symbol is of_symbol:
                break
            idx += symbol.size
        return idx

    def create_symbol(self, **kwargs):
        kwargs.update(dict(field_type=self))
        self._symbols.append(self.symbol_class(**kwargs))
        self._count += getattr(self._symbols[-1], 'size', 1)

    def create_dataclass(self):
        # TODO: do processing to handle different field types
        fields = [(symbol.name, float) for symbol in self._symbols]
        name = make_class_name([self._model_name, self._name])
        self._dataclass = make_dataclass(
            name,
            fields,
            bases=(FieldValues,),
            namespace=dict(asdict=dataclass_asdict),
        )

    def __iter__(self):
        for symbol in self._symbols:
            yield symbol

    def __len__(self):
        return len(self._symbols)

    def __getattr__(self, with_name):
        return self.get(name=with_name).backend_repr


def make_class_name(components):
    separate_words = ' '.join([comp.replace('_', ' ') for comp in components])
    # use pascal case from https://stackoverflow.com/a/8347192
    # TODO: this is turning mid-word uppercase (e.g., from a CamelCaseClassName) -- how
    # to just capitalize first character of word?
    return ''.join(word for word in separate_words.title() if not word.isspace())


@dataclass
class FrontendSymbolData:
    field_type: Field
    backend_repr: backend.symbol_class
    name: str = ''

    def flat_index(self):
        return self.field_type.flat_index(self)


@dataclass
class BaseSymbol(FrontendSymbolData, BackendSymbolData,):
    def __repr__(self):
        return f"<{self.field_type._resolve_name}: {self.name}>"


class IndependentSymbol(BaseSymbol):
    pass
    # TODO: long description?

class IndependentField(Field):
    """
    Mixin for fields which generate a symbol that are assigned to a class attribute
    during model definition.
    """

    pass


@dataclass(repr=False)
class FreeSymbol(IndependentSymbol):
    upper_bound: float = np.inf
    lower_bound: float = -np.inf
    # Then if bounds are here, must follow broadcasting rules

    def __post_init__(self):
        super().__post_init__()
        self.upper_bound = np.broadcast_to(self.upper_bound, self.shape)
        self.lower_bound = np.broadcast_to(self.lower_bound, self.shape)


class FreeField(IndependentField, default_direction=Direction.input):
    def make_backend_symbol(
        self, backend_name, shape=(1,), symmetric=False, diagonal=False, **kwargs
    ):
        if isinstance(shape, int):
            shape = (shape,)
        out = backend.symbol_generator(
            name=backend_name, shape=shape, symmetric=symmetric, diagonal=diagonal
        )
        symbol_data = backend.get_symbol_data(out)
        kwargs.update(
            backend_repr=out,
            **asdict(symbol_data),
        )
        return kwargs

    def __call__(self, **kwargs):
        backend_name = "%s_%d" % (self._resolve_name, len(self._symbols))
        new_kwargs = self.make_backend_symbol(backend_name=backend_name, **kwargs)
        self.create_symbol(**new_kwargs)
        return self._symbols[-1].backend_repr


@dataclass(repr=False)
class WithDefaultSymbol(FreeSymbol,):
    default: float = 0.   # TODO union[numeric, expression] or None?

class WithDefaultField(FreeField):
    pass

@dataclass(repr=False)
class InitializedSymbol(FreeSymbol,):
    initializer: float = 0.  # TODO union[numeric, expression]
    warm_start: bool = True

class InitializedField(FreeField):
    pass

@dataclass(repr=False)
class BoundedAssignmentSymbol(FreeSymbol):
    pass

class BoundedAssignmentField(Field, default_direction=Direction.output):
    def __call__(self, value, eq=None, name='', **kwargs):
        symbol_data = backend.get_symbol_data(value)
        if not name:
            name="%s_%s_%d" % (self._model_name, self._name, len(self._symbols))
        if eq is not None:
            if 'lower_bound' in kwargs or 'upper_bound' in kwargs:
                raise ValueError
            kwargs['lower_bound'] = eq
            kwargs['upper_bound'] = eq
        self.create_symbol(name=name, backend_repr=value,  **kwargs, **asdict(symbol_data))


@dataclass(repr=False)
class TrajectoryOutputSymbol(IndependentSymbol):
    terminal_term: BaseSymbol = 0.
    integrand: BaseSymbol = 0.

class TrajectoryOutputField(IndependentField, default_direction=Direction.output):
    def __call__(self, terminal_term=0, integrand=0, **kwargs):
        # Use quadrature instead of state.
        # to create a copy of state on TrajectoryAnalysis and give trajectory_output
        # access to new copy, would need to add a reference at copy time which is
        # doable but less clean, especially since this is just a hack to avoid
        # quadrature anyway
        # then didn't need to re-arrange ModelType.__new__ to give access to inner_to, 
        # etc. before implementation binding. 
        # TODO: undo inner_to refactor from  0195013ddf58dc1fa8f589d99671ba231ab846a6


        if isinstance(terminal_term, BaseSymbol):
            terminal_term = terminal_term.backend_repr
        if isinstance(integrand, BaseSymbol):
            integrand = integrand.backend_repr

        if isinstance(terminal_term, backend.symbol_class):
            # comstant terminal terms are handled below
            shape_data = backend.get_symbol_data(terminal_term)
            kwargs['terminal_term'] = terminal_term


        if isinstance(integrand, backend.symbol_class):
            if 'terminal_term' in kwargs:
                assert backend.get_symbol_data(integrand) == shape_data
            else:
                shape_data = backend.get_symbol_data(integrand)
                kwargs['terminal_term'] = np.broadcast_to(terminal_term, shape_data.shape)
        else:
            integrand = np.broadcast_to(integrand, shape_data.shape)

        kwargs['integrand'] = integrand

        shape_data_dict = asdict(shape_data)


        traj_out_placeholder = backend.symbol_generator(
            name=f'trajectory_output_{len(self._symbols)}',
            **shape_data_dict,
        )


        self.create_symbol(
            backend_repr=traj_out_placeholder,
            **kwargs,
            **shape_data_dict,
        )
        return traj_out_placeholder


class DependentSymbol(BaseSymbol):
    pass

@dataclass(repr=False)
class AssignedSymbol(BaseSymbol):
    pass

class AssignedField(Field, default_direction=Direction.output):

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            # TODO: resolve circular imports so we can use dataclass
            if isinstance(value, BaseSymbol):
                value = value.backend_repr
            symbol_data = backend.get_symbol_data(value)
            self.create_symbol(
                name=name,
                backend_repr=value,
                # TODO: I guess if this accepts model instances, it becomes recursive to
                # allow dot access to sub systems? Actually, this breaks the idea of
                # both system encapsolation and implementations. So don't do it, but
                # doument it. Can programatically add sub-system outputs though. For
                # these reasons, ditch intermediate stuff.
                **asdict(symbol_data)
            )
            if self._direction == Direction.output and self._cls_dict:
                self._cls_dict[name] = value
            #super().__setattr__(name, self._symbols[-1])



@dataclass(repr=False)
class MatchedSymbolMixin:
    match: BaseSymbol # match to the Symbol instance

    def update_name(self):
        self.name = '__'.join([self.field_type._name, self.match.name])

@dataclass(repr=False)
class MatchedSymbol(BaseSymbol, MatchedSymbolMixin,):
    pass


class MatchedField(Field,):
    def __init__(self, matched_to=None, **kwargs):
        """
        matched_to is Field instance that this MatchedField is matched to.
        """
        # TODO: add matches_required flag? e.g. for initializer
        # TODO: a flag for application direction? Field._diretion is relative to model;
        # matched (especially for model internal) could be used as a free or assigned
        # field (eg,dot for  DAE vs ODE), maybe "usage" and it can directly refer to
        # FreeField or AssignedField instead of a separate enum?
        # TODO: should FreeField instance __call__ get a kwarg for all matched fields
        # that reference it and are assigned? -> two ways of assigning the match...

        super().__init__(**kwargs)
        self._matched_to = matched_to
        self._init_kwargs.update(dict(matched_to=matched_to))

    def __setitem__(self, key, value):
        if isinstance(key, backend.symbol_class):
            match = self._matched_to.get(backend_repr=key)
            if isinstance(match, list):
                raise ValueError
        elif isinstance(key, BaseSymbol):
            match = key
        else:
            raise ValueError
        symbol_data = backend.get_symbol_data(value)
        self.create_symbol(
            name=None,
            match=match,
            backend_repr=value,
            **asdict(symbol_data)
        )

    def __getitem__(self, key):
        """
        get the matched symbodl by the matche's name, backend symbol, or symbol
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
            # TODO: could easily create a new symbol related to match; should it depend
            # on direction? Does matched need two other direction options for internal
            # get vs set? or is that a separate type of matchedfield?
            raise ValueError
        return item.backend_repr

