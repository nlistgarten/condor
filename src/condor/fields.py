
# TODO: figure out python version minimum

from dataclasses import dataclass, make_dataclass, asdict
import numpy as np
from enum import Enum
from condor.backends.default import backend
from condor.backends import BackendSymbolData

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
        # TODO: make sure python version is correct
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
class FrontendSymbolData:
    field_type: Field
    name: str
    backend_repr: backend.symbol_class


@dataclass
class BaseSymbol(BackendSymbolData, FrontendSymbolData):
    def __repr__(self):
        return f"<{self.field_type._resolve_name}: {self.name}>"

class IndependentSymbol(BaseSymbol):
    pass
    # TODO: long description?

@dataclass(repr=False)
class FreeSymbol(IndependentSymbol):
    upper_bound: float
    lower_bound: float
    # TODO: this data structure is  repeated in 2 other places, Symbol.__call__ and
    # symbol_generator. It should be dry-able, but validation is complicated..
    # one idea: where symbols gets instantiated, use **kwargs to try constructing,
    # Symbol class will at least validate field name assignment. Then Field
    # subclass does model-level validation.
    # or can the symbol itself perform validation? It has field_type back
    # referene, so it could do it...
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

    def __call__(
        self, shape=(1,), symmetric=False, diagonal=False,
            upper_bound=np.inf, lower_bound=np.inf
    ):
        if symmetric:
            raise NotImplemented
        if diagonal:
            raise NotImplemented

        if isinstance(shape, int):
            shape = (shape,)

        size = np.prod(shape)

        if diagonal:
            assert size == shape[0]
        elif symmetric:
            assert len(shape) == 2
            assert shape[0] == shape[1]
        pass_kwargs = dict(
            name="%s_%d" % (self._resolve_name, len(self._symbols)),
            shape=shape,
            symmetric=symmetric,
            diagonal=diagonal,
        )
        if symmetric:
            n = shape[0]
            size = int(n*(n+1)/2)
        self._count += size
        out = backend.symbol_generator(**pass_kwargs)
        symbol_data = backend.get_symbol_data(out)
        self.create_symbol(
            name=None,
            backend_repr=out,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            **asdict(symbol_data)
        )
        return out



@dataclass(repr=False)
class AssignedSymbol(BaseSymbol):
    pass

class AssignedField(Field, symbol_class=AssignedSymbol):
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
            # TODO: resolve circular imports so we can use dataclass
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
            super().__setattr__(name, self._symbols[-1])

@dataclass(repr=False)
class MatchedSymbol(BaseSymbol):
    match: BaseSymbol # match to the Symbol instance

    def update_name(self):
        self.name = '__'.join([self.field_type._name, self.match.name])

class MatchedField(Field, symbol_class=MatchedSymbol):
    def __init__(
        self, matched_to=None, direction=Direction.internal, name='', model=None, inherit_from=None
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
        return item

