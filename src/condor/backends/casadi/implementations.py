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

    def __call__(self, model_instance, *args):
        # TODO: can probably wrap then flatten instead of incomplete flatten?
        model_instance.bind_field(
            self.model.output, self.func(*args)
        )

class InitializerMixin:
    def set_initial(self, *args, **kwargs):
        if args:
            # assume it's all initialized values, so flatten and assign
            self.x0 = flatten(args)
        else:
            count_accumulator = 0
            for field in self.parsed_initialized_fields:
                size_cum_sum = np.cumsum([count_accumulator] + field.list_of('size'))
                for start_idx, end_idx, name in zip(
                    size_cum_sum, size_cum_sum[1:], field.list_of('name')
                ):
                    if name in kwargs:
                        self.x0[start_idx:end_idx] = flatten(kwargs[name])
                count_accumulator += field._count

    def parse_initializers(self, fields, initializer_args):
        # currently assuming all initializer fields get initialized together
        # TODO: flatten -- might be done?
        x0_at_construction = []
        initializer_exprs = []
        # TODO: shoud broadcasting be done elsewhere? with field? bounds needs it too

        if isinstance(fields, co.Field):
            fields = [fields]


        for field in fields:
            defined_initializers = field.list_of("initializer")
            for solver_var in field:
                if isinstance(solver_var.initializer, casadi.MX):
                    x0_at_construction.extend(np.zeros(solver_var.size))
                    initializer_exprs.extend(
                        casadi.vertsplit(solver_var.initializer.reshape((-1,1)))
                    )
                elif solver_var.warm_start:
                    x0_at_construction.extend(
                        np.broadcast_to(solver_var.initializer, (solver_var.size,))
                    )
                    initializer_exprs.extend(
                        casadi.vertsplit(solver_var.backend_repr.reshape((-1,1)))
                    )
                else:
                    x0_at_construction.append(solver_var.initializer)
                    initializer_exprs.append(solver_var.initializer)

        self.parsed_initialized_fields = fields

        #setattr(self, f"{field._name}_at_construction", flatten(x0_at_construction))
        self.initial_at_construction = flatten(x0_at_construction)
        initializer_exprs = flatten(initializer_exprs)
        #setattr(self, f"{field._name}_initializer_func",
        self.initializer_func = casadi.Function(
            f"{field._model_name}_{field._name}_initializer",
            initializer_args,
            initializer_exprs,
        )


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
            #self.implicit_output_at_construction,
            self.initial_at_construction,
            rootfinder_options,
            self.initializer_func,
        )

    def __call__(self, model_instance, *args, **kwargs):
        out = self.callback(casadi.vertcat(*flatten(args)))
        model_instance.bind_field(
            self.model.implicit_output,
            out[:self.model.implicit_output._count]
        )
        model_instance.bind_field(
            self.model.explicit_output,
            out[:self.model.explicit_output._count]
        )

        if hasattr(self.callback, 'resid'):
            # TODO: verify this will never become stale
            # TODO: it seems a bit WET to wrap each field and then, eventually, bind
            # each field. 
            resid = self.callback.resid
            model_instance.bind_field(
                self.model.residual, resid, symbols_to_instance=False
            )

    def set_initial(self, *args, **kwargs):
        self.x0 = self.callback.x0
        super().set_initial(*args, **kwargs)
        self.callback.x0 = self.x0

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

        self.x0 = self.initial_at_construction.copy()
        #self.variable_at_construction.copy()

        self.optimizer = casadi.nlpsol(
            model.__name__,
            "ipopt",
            self.nlp_args,
            self.ipopt_opts,
        )
        # additional options from https://groups.google.com/g/casadi-users/c/OdRQKR13R50/m/bIbNoEHVBAAJ
        # to try to get sensitivity from ipopt. so far no...

    def __call__(self, model_instance, *args):
        initializer_args = [self.x0]
        if self.has_p:
            p = casadi.vertcat(*flatten(args))
            initializer_args += [p]
        if not self.has_p or not isinstance(args[0], casadi.MX):
            self.x0 = casadi.vertcat(
                #*self.variable_initializer_func(*initializer_args)
                *self.initializer_func(*initializer_args)
            ).toarray().reshape(-1)
        call_args = dict(
            x0=self.x0, ubx=self.ubx, lbx=self.lbx, ubg=self.ubg, lbg=self.lbg
        )

        out = self.optimizer(**call_args)
        if not self.has_p or not isinstance(args[0], casadi.MX):
            self.x0 = out["x"]

        model_instance.bind_field(self.model.variable, out["x"])
        model_instance.bind_field(self.model.constraint, out["g"])
        model_instance.objective = out["f"].toarray()[0,0]

