import numpy as np

from condor.backend import callables_to_operator, expression_to_operator

from .utils import options_to_kwargs


class DeferredSystem:
    def construct(self, model):
        self.symbol_inputs = model.input.flatten()
        self.symbol_outputs = model.output.flatten()
        self.model = model

    def __init__(self, model_instance):
        self.construct(model_instance.__class__)
        self(model_instance)

    def __call__(self, model_instance):
        model_instance.bind_field(self.model.output.wrap(self.symbol_outputs))


class ExplicitSystem:
    """Implementation for :class:`ExplicitSystem` model.

    No :class:`Options` expected.
    """

    def __init__(self, model_instance):
        self.construct(model_instance.__class__)
        self(model_instance)

    def construct(self, model):
        self.symbol_inputs = model.input.flatten()
        self.symbol_outputs = model.output.flatten()

        self.model = model
        self.func = expression_to_operator(
            [self.symbol_inputs],
            self.symbol_outputs,
            name=model.__name__,
        )

    def __call__(self, model_instance):
        self.args = model_instance.input.flatten()
        self.out = self.func(self.args)
        model_instance.bind_field(self.model.output.wrap(self.out))


class ExternalSolverModel:
    """Implementation for External Solver models.

    No :class:`Options` expected.
    """

    def __init__(self, model_instance):
        model = model_instance.__class__
        model_instance.options_dict = options_to_kwargs(model)
        self.construct(model, **model_instance.options_dict)
        self(model_instance)

    def construct(self, model):
        self.model = model
        self.wrapper = model._meta.external_wrapper
        self.input = model.input.flatten()
        self.output = model.output.flatten()

        wrapper_funcs = [self._wrap_user_func(self.wrapper.function)]
        if hasattr(self.wrapper, "jacobian"):
            wrapper_funcs.append(self._wrap_user_func(self.wrapper.jacobian, der=1))
        if hasattr(self.wrapper, "hessian"):
            wrapper_funcs.append(self._wrap_user_func(self.wrapper.hessian, der=2))

        self.callback = callables_to_operator(
            wrapper_funcs,
            self,
            jacobian_of=None,
            input_symbol=self.input,
            output_symbol=self.output,
        )
        self.callback.construct()

    def __call__(self, model_instance):
        use_args = model_instance.input.flatten()
        out = self.callback(use_args)
        model_instance.bind_field(self.model.output.wrap(out))

    def _wrap_user_func(self, user_func, der=0):
        def func(inputs):
            input_data = self.model.input.wrap(inputs)
            user_out = user_func(input_data)
            # TODO: normalize user_out with fields (e.g., output.wrap,) but then need
            # jacobian, hessian fields to work...
            if isinstance(user_out, dict):
                # pretty sure there's a itertools way to generalize this...
                if der == 0:
                    out = np.concatenate(
                        [
                            np.ravel(user_out[n])
                            for n in self.model.output.list_of("name")
                        ]
                    )
                elif der == 1:
                    out = np.concatenate(
                        [
                            np.ravel(user_out[no, ni])
                            for no in self.model.output.list_of("name")
                            for ni in self.model.input.list_of("name")
                        ]
                    )

            else:
                out = user_out
            return out

        return func
