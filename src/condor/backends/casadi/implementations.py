import casadi
import condor as co
import numpy as np

def flatten(symbols):
    if isinstance(symbols, co.Field) or isinstance(symbols[0], casadi.MX):
        # imp construction or symbolic
        return [
            elem
            for symbol in symbols
            for col in casadi.horzsplit(symbol.backend_repr)
            for elem in casadi.vertsplit(col)
        ]
    else:
        # TODO: confirm that this reshape is the same
        return [
            elem
            for symbol in symbols
            for elem in np.atleast_1d(symbol).reshape(-1)
        ]

def wrap(field, values):
    size_cum_sum = np.cumsum([0] + field.list_of('size'))
    # TODO: need to make sure 
    values = np.atleast_1d(values).reshape(-1)
    return tuple([
        values[start_idx:end_idx]
        for start_idx, end_idx in zip(size_cum_sum, size_cum_sum[1:])
    ])


class ExplicitSystem:
    def __init__(self, model):
        symbol_inputs = flatten(model.input)
        symbol_outputs = flatten(model.output)
        name_inputs = model.input.list_of('name')
        name_outputs = model.output.list_of('name')
        self.model = model
        doc = f"{tuple(name_outputs)} = {model}{tuple(name_outputs)}"
        self.func =  casadi.Function(model.__name__, symbol_inputs, symbol_outputs)

    def __call__(self, *args):
        out = self.func(*flatten(args))
        wrapped = wrap(self.model.output, out)
        return wrapped


class AlgebraicSystem:
    #  actually most of these should be classes? at least attach model

    #bound_behavior = enum(
    #    ignore,
    #    # I think backtracking behavior can be referred to as L2 for "vector" and L1 for
    #    # "scalar"  
    #)
    #
    #line_search_criteria = enum(
    ## wolf,?
    #    # or just true or false? 
    #)
    #
    def __init__(
        model, atol=1E-12, rtol=1E-12, warm_start=True, exact_hessian=True,
    ):
        return
        rootfinder_options = dict(
        )
        self.x = casadi.vertcat(*[
            flatten(out) for out in model.implicit_output.list_of('symbol')
        ])
        self.g0 = casadi.vertcat(*[
            flatten(out) for out in model.residual.list_of('symbol')
        ])
        self.g1 = casadi.vertcat(*[
            flatten(out) for out in model.explicit_output.list_of('symbol')
        ])
        self.p = casadi.vertcat(*[
            flatten(out) for out in model.input.list_of('symbol')
        ])

        self.model = model
        self.rootfinder = casadi.Rootfinder()

    def __call__(self, **kwargs):
        p_arg = casadi.vertcat(args)


