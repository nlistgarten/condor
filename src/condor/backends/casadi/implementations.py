import casadi
import condor as co
import numpy as np
from enum import Enum, auto
from condor.backends.casadi.utils import flatten, wrap
from condor.backends.casadi.algebraic_solver import SolverWithWarmStart
from condor.backends.casadi.shooting_gradient_method import ShootingGradientMethod


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
        # bounds are definitely done in FreeField, I believe this reshape/broadcast_to
        # is complete (just for initializer attribute on initialized field) -- I guess
        # it might be drier in post_init of initialized field?

        if isinstance(fields, co.Field):
            fields = [fields]


        for field in fields:
            defined_initializers = field.list_of("initializer")
            for solver_var in field:
                if isinstance(solver_var.initializer, casadi.MX):
                    x0_at_construction.extend(np.zeros(solver_var.size))
                    initializer_exprs.extend(
                        casadi.vertsplit(
                            solver_var.initializer.reshape((solver_var.size,1))
                        )
                    )
                else:
                    shaped_initial = np.broadcast_to(
                        solver_var.initializer, solver_var.shape
                    ).reshape(-1)
                    if solver_var.warm_start:
                        x0_at_construction.extend(shaped_initial)
                        initializer_exprs.extend(
                            casadi.vertsplit(solver_var.backend_repr.reshape((-1,1)))
                        )
                    else:
                        x0_at_construction.append(shaped_initial)
                        initializer_exprs.append(shaped_initial)

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

def get_state_setter(field, setter_args, default=0.):
    """
    field is MatchedField matching to the state
    """
    setter_exprs = []
    if isinstance(setter_args, casadi.MX):
        setter_args = [setter_args]

    for state in field._matched_to:
        setter_symbol = field.get(match=state)
        if default is None:
            default = state.backend_repr
        if isinstance(setter_symbol, list) and len(setter_symbol) == 0:
            setter_symbol = default
        elif isinstance(setter_symbol, co.BaseSymbol):
            setter_symbol = setter_symbol.backend_repr
        else:
            raise ValueError

        if isinstance(setter_symbol, casadi.MX):
            setter_exprs.extend(
                casadi.vertsplit(setter_symbol.reshape((state.size,1)))
            )
        else:
            # casadi.MX is always a matrix, and broadcast_to cannot handle (n,) to (n,1)
            if state.shape[1] == 1:
                setter_exprs.append(
                    np.broadcast_to(setter_symbol, state.size).reshape(-1)
                )
            else:
                setter_exprs.append(
                    np.broadcast_to(setter_symbol, state.shape).reshape(-1)
                )
    #setter_exprs = flatten(setter_exprs)
    setter_exprs = [casadi.vertcat(*setter_exprs)]
    setter_func = casadi.Function(
        f"{field._model_name}_{field._matched_to._name}_{field._name}",
        setter_args,
        setter_exprs,
    )
    setter_func.expr = setter_exprs[0]
    return setter_func



class TrajectoryAnalysis:
    def __init__(self, model):
        self.model = model
        self.ode_model = ode_model = model.inner_to

        self.p = casadi.vertcat(*flatten(ode_model.parameter))
        self.x = casadi.vertcat(*flatten(ode_model.state))
        self.simulation_signature = [
            self.p,
            self.ode_model.independent_variable,
            self.x,
        ]
        self.sym_event_channel = casadi.MX.sym('event_channel')


        self.traj_out = casadi.vertcat(*flatten(model.trajectory_output))
        self.traj_out_func = casadi.Function(
            f"{model.__name__}_trajectory_output",
            self.simulation_signature,
            [self.traj_out]
        )

        self.x0 = get_state_setter(ode_model.initial, self.p)
        self.y_expr = casadi.vertcat(*flatten(ode_model.output))
        self.e_expr = [event.function for event in ode_model.Event.subclasses]
        self.h_expr = sum([
            get_state_setter(
                event.update,
                self.simulation_signature
            ).expr*(self.sym_event_channel==idx)
            for idx, event in enumerate(ode_model.Event.subclasses)
        ])


        self.simupy_shape_data = dict(
            dim_state = ode_model.state._count,
            dim_output = ode_model.output._count,
        )
        self.simupy_func_kwargs = dict(
            state_equation_function = get_state_setter(
                ode_model.dot,
                self.simulation_signature
            ),
            output_equation_function = casadi.Function(
                f"{ode_model.__name__}_output",
                self.simulation_signature,
                [self.y_expr],
            ),
            event_equation_function = casadi.Function(
                f"{ode_model.__name__}_event",
                self.simulation_signature,
                self.e_expr,
            ),
            update_equation_function = casadi.Function(
                f"{ode_model.__name__}_update",
                self.simulation_signature + [self.sym_event_channel],
                [self.h_expr],
            ),

        )
        self.callback = ShootingGradientMethod(self)

