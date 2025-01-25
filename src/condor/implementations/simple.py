from .utils import options_to_kwargs
import condor as co
import numpy as np
#from co.backend.utils import (flatten,  
#                                            vertcat)
from condor import backend
import casadi

flatten = co.backend.utils.flatten
vertcat = co.backend.utils.vertcat
FunctionsToOperator = co.backend.utils.FunctionsToOperator


class DeferredSystem:
    def construct(self, model):
        self.symbol_inputs = model.input.flatten()
        self.symbol_outputs = model.output.flatten()

        self.model = model
        self.func = casadi.Function(
            model.__name__,
            self.symbol_inputs,
            self.symbol_outputs,
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
            self.symbol_outputs,
        )


class ExplicitSystem:
    """ Implementation for :class:`ExplicitSystem` model. No :class:`Options` expected.
    """

    def __init__(self, model_instance, args):
        self.construct(model_instance.__class__)
        self(model_instance, *args)

    def construct(self, model):
        self.symbol_inputs = model.input.flatten()
        self.symbol_outputs = model.output.flatten()

        self.model = model
        self.func = casadi.Function(
            model.__name__,
            [self.symbol_inputs],
            [self.symbol_outputs],
            dict(
                allow_free=True,
            ),
        )

    def __call__(self, model_instance, *args):
        self.args = model_instance.input.flatten()
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
        self.input = model.input.flatten()
        self.output = model.output.flatten()
        # self.input = model.input.list_of('backend_repr')
        # self.output = model.output.list_of('backend_repr')
        wrapper_funcs = [self.wrapper.function]
        if hasattr(self.wrapper, "jacobian"):
            wrapper_funcs.append(self.wrapper.jacobian)
        if hasattr(self.wrapper, "hessian"):
            wrapper_funcs.append(self.wrapper.hessian)
        self.callback = FunctionsToOperator(
            wrapper_funcs, self, jacobian_of=None,
            input_symbol = self.input,
            output_symbol = self.output,
        )
        self.callback.construct()


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

