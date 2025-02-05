from .utils import options_to_kwargs
import condor as co
import numpy as np
from condor import backend

from condor.backend import callables_to_operator, expression_to_operator
class DeferredSystem:
    def construct(self, model):
        self.symbol_inputs = model.input.flatten()
        self.symbol_outputs = model.output.flatten()
        self.model = model

    def __init__(self, model_instance):
        self.construct(model_instance.__class__)
        self(model_instance)

    def __call__(self, model_instance):
        model_instance.bind_field(
            self.model.output.wrap(self.symbol_outputs)
        )


class ExplicitSystem:
    """ Implementation for :class:`ExplicitSystem` model. No :class:`Options` expected.
    """

    def __init__(self, model_instance):
        self.construct(model_instance.__class__)
        self(model_instance)

    def construct(self, model):
        self.symbol_inputs = model.input.flatten()
        self.symbol_outputs = model.output.flatten()

        self.model = model
        self.func = expression_to_operator(
            self.symbol_inputs,
            self.symbol_outputs,
            name=model.__name__,
        )

    def __call__(self, model_instance):
        self.args = model_instance.input.flatten()
        self.out = self.func(self.args)
        model_instance.bind_field(
            self.model.output.wrap(self.out)
        )

class ExternalSolverModel:
    def __init__(self, model_instance):
        model = model_instance.__class__
        self.construct(model, **options_to_kwargs(model))
        self(model_instance)

    def construct(self, model):
        self.model = model
        self.wrapper = model._meta.external_wrapper
        self.input = model.input.flatten()
        self.output = model.output.flatten()
        wrapper_funcs = [self.wrapper.function]
        if hasattr(self.wrapper, "jacobian"):
            wrapper_funcs.append(self.wrapper.jacobian)
        if hasattr(self.wrapper, "hessian"):
            wrapper_funcs.append(self.wrapper.hessian)
        self.callback = callables_to_operator(
            wrapper_funcs, self, jacobian_of=None,
            input_symbol = self.input,
            output_symbol = self.output,
        )
        self.callback.construct()


    def __call__(self, model_instance):
        use_args = model_instance.input.flatten()
        out = self.callback(use_args)
        model_instance.bind_field(
            self.model.output.wrap(out)
        )

