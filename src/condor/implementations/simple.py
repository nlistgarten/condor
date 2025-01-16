from .utils import options_to_kwargs
from condor.backends.casadi.table_lookup import NDSplinesCallback
import condor as co
import numpy as np
#from co.backend.utils import (flatten,  
#                                            vertcat)
from condor import backend

flatten = co.backend.utils.flatten
vertcat = co.backend.utils.flatten

class DeferredSystem:
    def construct(self, model):
        symbol_inputs = model.input.list_of("backend_repr")
        symbol_outputs = model.output.list_of("backend_repr")

        self.symbol_inputs = [casadi.vertcat(*flatten(symbol_inputs))]
        self.symbol_outputs = [casadi.vertcat(*flatten(symbol_outputs))]

        name_inputs = model.input.list_of("name")
        name_outputs = model.output.list_of("name")
        self.model = model
        self.func = casadi.Function(
            model.__name__,
            symbol_inputs,
            symbol_outputs,
            dict(
                allow_free=True,
            ),
        )

    def __init__(self, model_instance, args):
        self.construct(model_instance.__class__)
        self(model_instance, *args)

    def __call__(self, model_instance, *args):
        model_instance.bind_field(
            self.model.output,
            self.symbol_outputs[0],
        )


class ExplicitSystem:
    """ Implementation for :class:`ExplicitSystem` model. No :class:`Options` expected.
    """

    def __init__(self, model_instance, args):
        self.construct(model_instance.__class__)
        self(model_instance, *args)

    def construct(self, model):
        symbol_inputs = model.input.list_of("backend_repr")
        symbol_outputs = model.output.list_of("backend_repr")

        self.symbol_inputs = [casadi.vertcat(*flatten(symbol_inputs))]
        self.symbol_outputs = [casadi.vertcat(*flatten(symbol_outputs))]

        name_inputs = model.input.list_of("name")
        name_outputs = model.output.list_of("name")
        self.model = model
        # self.func =  casadi.Function(model.__name__, self.symbol_inputs, self.symbol_outputs)
        self.func = casadi.Function(
            model.__name__,
            self.symbol_inputs,
            self.symbol_outputs,
            dict(
                allow_free=True,
            ),
        )

    def __call__(self, model_instance, *args):
        self.args = casadi.vertcat(*flatten(args))
        self.out = self.func(self.args)
        model_instance.bind_field(
            self.model.output,
            self.out,
        )

class ExternalSolverModel:
    def __init__(self, model_instance, args):
        model = model_instance.__class__
        self.construct(model, **options_to_kwargs(model))
        self(model_instance, *args)

    def construct(self, model):
        self.model = model
        self.wrapper = model._meta.external_wrapper
        self.input = casadi.vertcat(*flatten(model.input))
        self.output = casadi.vertcat(*flatten(model.output))
        # self.input = model.input.list_of('backend_repr')
        # self.output = model.output.list_of('backend_repr')
        self.placeholder_func = casadi.Function(
            f"{model.__name__}_placeholder",
            [self.input],
            [self.output],
            # self.input,
            # self.output,
            dict(
                allow_free=True,
            ),
        )
        self.callback = CasadiFunctionCallback(
            self.placeholder_func, self.wrapper.function, self, jacobian_of=None
        )
        if hasattr(self.wrapper, "jacobian"):
            self.callback.jacobian = CasadiFunctionCallback(
                self.placeholder_func.jacobian(),
                self.wrapper.jacobian,
                implementation=self,
                jacobian_of=self.callback,
            )

            self.callback.jacobian.construct(
                self.callback.jacobian.placeholder_func.name(),
                self.callback.jacobian.opts,
            )
        self.callback.construct(
            self.callback.placeholder_func.name(), self.callback.opts
        )

    def __call__(self, model_instance, *args):
        use_args = casadi.vertcat(*flatten(args))
        out = self.callback(use_args)
        # out = self.callback(*args)
        if isinstance(out, casadi.MX):
            out = casadi.vertsplit(out)
        model_instance.bind_field(
            self.model.output,
            out,
        )

class TableLookup:
    """ Implementation for :class:`TableLookup` model.
    Should be re-factored to use ExternalSolver representation
    """
    def __init__(self, model_instance, args):
        model = model_instance.__class__
        self.construct(model, **options_to_kwargs(model))
        self(model_instance, *args)

    def construct(
        self,
        model,
        degrees=3,
        bcs=(-1,0),
    ):
        """ Pass through 
        """
        self.model = model
        self.symbol_inputs = model.input.list_of("backend_repr")
        self.symbol_outputs = model.output.list_of("backend_repr")
        self.input_data = [
            model.input_data.get(match=inp).backend_repr for inp in model.input
        ]
        self.output_data = np.stack(
            [model.output_data.get(match=out).backend_repr for out in model.output],
            axis=-1,
        )
        self.degrees = degrees
        self.bcs = bcs
        self.callback = NDSplinesCallback(self)

    def __call__(self, model_instance, *args):
        out = self.callback(casadi.vertcat(*flatten(args)))
        if isinstance(out, casadi.MX):
            out = casadi.vertsplit(out)
        model_instance.bind_field(
            self.model.output,
            out,
        )

