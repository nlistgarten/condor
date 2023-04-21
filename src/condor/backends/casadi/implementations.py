import casadi
import condor as co
import numpy as np
from enum import Enum, auto
from condor.backends.casadi import utils, algebraic_solver

def flatten(symbols):
    if isinstance(symbols, co.Field):
        symbols = symbols.list_of("backend_repr")

    if not symbols:
        return symbols

    if isinstance(symbols[0], casadi.MX):
        # imp construction or symbolic
        return [
            elem
            for symbol in symbols
            for col in casadi.horzsplit(symbol)
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


class SolverWithWarmStart(utils.CasadiFunctionCallbackMixin, casadi.Callback):

    def __init__(self, name, x, p, g0, g1, lbx, ubx, x0, rootfinder_options, initializer):
        casadi.Callback.__init__(self)
        self.name = name
        self.initializer = initializer
        self.x0 = x0
        self.lbx = np.array(lbx)
        self.ubx = np.array(ubx)

        self.newton = algebraic_solver.Newton(
            x,
            p,
            g0,
            lbx=lbx,
            ubx=ubx,
            tol=1e-10,
            ls_type=None,
            max_iter=100,
        )

        self.resid_func = casadi.Function(f"{name}_resid_func", [x, p], [g0, g1])

        self.rootfinder = casadi.rootfinder(
            f"{name}_rootfinder",
            "newton",
            self.resid_func,
            rootfinder_options,
        )
        out_imp, out_exp = self.rootfinder(self.x0, p)

        self.func = casadi.Function(
            f"{name}_rootfinder_func",
            casadi.vertsplit(p),
            casadi.vertsplit(out_imp) + casadi.vertsplit(out_exp),
        )

        self.construct(name, {})

    def eval(self, args):
        p = casadi.vertcat(*args)
        self.x0 = self.initializer(self.x0, p)

        self.x0 = np.array(self.x0).reshape(-1)
        p = p.toarray().reshape(-1)

        lbx_violations = np.where(self.x0 < self.lbx)[0]
        ubx_violations = np.where(self.x0 > self.ubx)[0]
        self.x0[lbx_violations] = self.lbx[lbx_violations]
        self.x0[ubx_violations] = self.ubx[ubx_violations]

        self.x0 = self.newton(self.x0, p)
        resid, out_exp = self.resid_func(self.x0, p)
        out_exp = out_exp.toarray().reshape(-1)

        return tuple([*self.x0, *out_exp])

