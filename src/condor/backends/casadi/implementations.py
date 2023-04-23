import casadi
import condor as co
import numpy as np
from enum import Enum, auto
from condor.backends.casadi.utils import flatten
from condor.backends.casadi.algebraic_solver import SolverWithWarmStart


class ExplicitSystem:
    def __init__(self, model):
        symbol_inputs = model.input.list_of("backend_repr")
        symbol_outputs = model.output.list_of("backend_repr")
        name_inputs = model.input.list_of('name')
        name_outputs = model.output.list_of('name')
        self.model = model
        self.func =  casadi.Function(model.__name__, symbol_inputs, symbol_outputs)

    def __call__(self, *args):
        return self.func(*args)


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
        re_initialize=slice(None), # slice for re-initializing at every call. will support indexing 
        default_initializer=0.,
        bound_behavior=BoundBehavior.p1,
        line_search_contraction = 0.5, # 1.0 -> no line search?
        line_search_criteria=LineSearchCriteria.armijo,
        eror_on_failure=False,
    ):
        rootfinder_options = dict(
        )
        self.x = casadi.vertcat(*flatten(model.implicit_output))
        self.g0 = casadi.vertcat(*flatten(model.residual))
        self.g1 = casadi.vertcat(*flatten(model.explicit_output))
        self.p = casadi.vertcat(*flatten(model.parameter))

        defined_initializers = model.implicit_output.list_of("initializer")
        x0_at_construction = []
        initializer_exprs = []
        for solver_var in model.implicit_output:
            if isinstance(solver_var.initializer, casadi.MX):
                x0_at_construction.append(np.zeros(solver_var.shape))
                initializer_exprs.append(solver_var.initializer)
            elif solver_var.warm_start:
                x0_at_construction.append(solver_var.initializer)
                initializer_exprs.append(solver_var.backend_repr)
            else:
                x0_at_construction.append(solver_var.initializer)
                initializer_exprs.append(solver_var.initializer)
        self.x0_at_construction = flatten(x0_at_construction)
        initializer_exprs = flatten(initializer_exprs)
        self.initializer_func = casadi.Function(
            f"{model.__name__}_initializer",
            [self.x, self.p],
            initializer_exprs,
        )

        self.model = model

        self.callback = SolverWithWarmStart(
            model.__name__,
            self.x,
            self.p,
            self.g0,
            self.g1,
            flatten(model.implicit_output.list_of("lower_bound")),
            flatten(model.implicit_output.list_of("upper_bound")),
            self.x0_at_construction,
            rootfinder_options,
            self.initializer_func,
        )

    def __call__(self, *args, **kwargs):
        return self.callback(*args)

class OptimizationProblem:
    def __init__(self, model):
        self.model = model
        self.objective = getattr(model, 'objective', 0)
        self.variables = casadi.vertcat(*flatten(model.variable))




