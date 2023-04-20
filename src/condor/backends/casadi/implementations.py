import casadi
import condor as co
import numpy as np
from enum import Enum, auto
import condor.backends.casadi.algebraic_solver as algebraic_solver

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
        self.func =  casadi.Function(model.__name__, symbol_inputs, symbol_outputs)

    def __call__(self, *args):
        out = self.func(*flatten(args))
        wrapped = wrap(self.model.output, out)
        return wrapped

class AlgebraicSystem:

    class BoundBehavior(Enum):
        p1 = auto() # enforce in a p1 sense ("scalar")
        p2 = auto() # enforce in a p2 sense ("vector")
        p1_hold = auto() # enforce in a p1 sense, modify search direction to hold enforced ("wall")
        ignore = auto()

    class LineSearchCriteria(Enum):
        armijo = auto()


    def __init__(
        self, model,
        # standard options
        atol=1E-12, rtol=1E-12, warm_start=True, exact_hessian=True, max_iters=100,
        # new options
        re_initialize=slice(None), # slice for re-initializing at every call
        default_initializer=0.,
        bound_behavior=AlgebraicSystem.BoundBehavior.p1,
        line_search_contraction = 0.5, # 1.0 -> no line search?
        line_search_criteria=AlgebraicSystem.LineSearchCriteria.armijo,
        eror_on_failure=False,
    ):
        rootfinder_options = dict(
        )
        self.x = casadi.vertcat(*flatten(model.implicit_output))
        self.g0 = casadi.vertcat(*flatten(model.residual))
        self.g1 = casadi.vertcat(*flatten(model.explicit_output))
        self.p = casadi.vertcat(*flatten(model.parameter))

        defined_initializers = model.initializer.list_of('match')
        x0_list = []
        for solver_var in model.implicit_output:
            if solver_var in defined_initializers:
                pass # check if it's an expression or a constant


        self.model = model
        self.rootfinder_func = casadi.Function(
            f"{model.__name__}_rootfunc",
            [self.x, self.p],
            [self.g0, self.g1],
        )
        #algebraic_solver.SolverWithWarmStart(
        #    dict(x=self.x, p=self.p),
        #    g_impl = self.g0,
        #    g_expl = self.g1,
        #)


    def __call__(self, **kwargs):
        p_arg = casadi.vertcat(args)


