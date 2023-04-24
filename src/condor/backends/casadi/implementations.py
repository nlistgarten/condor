import casadi
import condor as co
import numpy as np
from enum import Enum, auto
from condor.backends.casadi.utils import flatten, wrap
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

class InitializerMixin:
    def parse_initializers(self, field, initializer_args):
        # TODO: flatten
        defined_initializers = field.list_of("initializer")
        x0_at_construction = []
        initializer_exprs = []
        for solver_var in field:
            if isinstance(solver_var.initializer, casadi.MX):
                x0_at_construction.append(np.zeros(solver_var.shape))
                initializer_exprs.append(solver_var.initializer)
            elif solver_var.warm_start:
                x0_at_construction.append(solver_var.initializer)
                initializer_exprs.append(solver_var.backend_repr)
            else:
                x0_at_construction.append(solver_var.initializer)
                initializer_exprs.append(solver_var.initializer)
        setattr(self, f"{field._name}_at_construction", flatten(x0_at_construction))
        initializer_exprs = flatten(initializer_exprs)
        setattr(self, f"{field._name}_initializer_func", casadi.Function(
            f"{field._model_name}_{field._name}_initializer",
            initializer_args,
            initializer_exprs,
        ))

class AlgebraicSystem(InitializerMixin):

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


        self.model = model
        self.parse_initializers(model.implicit_output, [self.x, self.p],)

        self.callback = SolverWithWarmStart(
            model.__name__,
            self.x,
            self.p,
            self.g0,
            self.g1,
            flatten(model.implicit_output.list_of("lower_bound")),
            flatten(model.implicit_output.list_of("upper_bound")),
            self.implicit_output_at_construction,
            rootfinder_options,
            self.implicit_output_initializer_func,
        )

    def __call__(self, *args, **kwargs):
        out = self.callback(*flatten(args))
        #out = self.callback(*args)
        wrap_implicit = wrap(
            self.model.implicit_output,
            out[:self.model.implicit_output._count]
        )
        wrap_explicit = wrap(
            self.model.explicit_output,
            out[:self.model.explicit_output._count]
        )
        return *wrap_implicit, *wrap_explicit

class OptimizationProblem(InitializerMixin):
    def __init__(
        self, model,
        exact_hessian=True, 
    ):
        self.model = model

        self.f = f = getattr(model, 'objective', 0)
        self.has_p = bool(len(model.parameter))
        self.p = p = casadi.vertcat(*flatten(model.parameter))
        self.x = x = casadi.vertcat(*flatten(model.variable))
        self.g = g = casadi.vertcat(*flatten(model.constraint))

        self.lbx = lbx = flatten(model.variable.list_of("lower_bound"))
        self.ubx = ubx = flatten(model.variable.list_of("upper_bound"))
        self.lbg = lbg = flatten(model.constraint.list_of("lower_bound"))
        self.ubg = ubg = flatten(model.constraint.list_of("upper_bound"))

        self.nlp_args = dict(f=f, x=x, g=g)
        initializer_args = [self.x]
        if self.has_p:
            initializer_args += [self.p]
            self.nlp_args["p"] = p
        self.parse_initializers(model.variable, initializer_args)
        self.ipopt_opts = ipopt_opts = dict(
            print_time= False,
            ipopt = dict(
                # hessian_approximation="limited-memory",
                print_level=5,  # 0-2: nothing, 3-4: summary, 5: iter table (default)
                tol=1E-14,
                #max_iter=1000,
            ),
            bound_consistency=True,
            #clip_inactive_lam=True,
            #calc_lam_x=False,
            #calc_lam_p=False,
        )

        self.x0 = self.variable_at_construction.copy()

        self.optimizer = casadi.nlpsol(
            model.__name__,
            "ipopt",
            self.nlp_args,
            self.ipopt_opts,
        )
        # additional options from https://groups.google.com/g/casadi-users/c/OdRQKR13R50/m/bIbNoEHVBAAJ
        # to try to get sensitivity from ipopt. so far no...

    def __call__(self, *args):
        initializer_args = [self.x0]
        if self.has_p:
            p = casadi.vertcat(*flatten(args))
            initializer_args += [p]
        if not self.has_p or not isinstance(args[0], casadi.MX):
            self.x0 = casadi.vertcat(
                *self.variable_initializer_func(*initializer_args)
            ).toarray().reshape(-1)
        call_args = dict(
            x0=self.x0, ubx=self.ubx, lbx=self.lbx, ubg=self.ubg, lbg=self.lbg
        )

        out = self.optimizer(**call_args)
        if not self.has_p or not isinstance(args[0], casadi.MX):
            self.x0 = out["x"]

        return wrap(self.model.variable, out["x"])


