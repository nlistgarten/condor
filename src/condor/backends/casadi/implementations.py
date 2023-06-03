import casadi
import condor as co
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from condor.backends.casadi.utils import flatten, wrap, symbol_class
from condor.backends.casadi.algebraic_solver import SolverWithWarmStart
from condor.backends.casadi.shooting_gradient_method import ShootingGradientMethod

from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

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
                if isinstance(solver_var.initializer, symbol_class):
                    x0_at_construction.extend(np.zeros(solver_var.size))
                    initializer_exprs.extend(
                        casadi.vertsplit(
                            solver_var.initializer.reshape((solver_var.size,1))
                        )
                    )
                else:
                    initializer_val = np.array(solver_var.initializer)
                    if initializer_val.size == solver_var.size:
                        shaped_initial = initializer_val.reshape(-1)
                    else:
                        shaped_initial = np.broadcast_to(
                            initializer_val, solver_var.shape
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
            out[self.model.implicit_output._count:self.model.implicit_output._count+self.model.explicit_output._count]
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

    class Method(Enum):
        ipopt = auto()
        scipy_cg = auto()
        scipy_trust_constr = auto()
        scipy_slsqp = auto()

    def __init__(
        self, model,
        exact_hessian=True,
        method=Method.ipopt
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
        self.x0 = self.initial_at_construction.copy()
        #self.variable_at_construction.copy()

        self.method = method
        if self.method is OptimizationProblem.Method.ipopt:

            self.ipopt_opts = ipopt_opts = dict(
                print_time= False,
                ipopt = dict(
                    print_level=5,  # 0-2: nothing, 3-4: summary, 5: iter table (default)
                    #tol=1E-14, # tighter tol for sensitivty
                    #accept_every_trial_step="yes",
                    #max_iter=1000,
                ),
                bound_consistency=True,
                #clip_inactive_lam=True,
                #calc_lam_x=False,
                #calc_lam_p=False,
            )
            # additional options from https://groups.google.com/g/casadi-users/c/OdRQKR13R50/m/bIbNoEHVBAAJ
            # to try to get sensitivity from ipopt. so far no...
            if not exact_hessian:
                ipopt_opts["ipopt"].update(
                    hessian_approximation="limited-memory",
                )

            self.optimizer = casadi.nlpsol(
                model.__name__,
                "ipopt",
                self.nlp_args,
                self.ipopt_opts,
            )
        else:
            self.optimizer = None
            self.f_func = casadi.Function(
                f"{model.__name__}_objective",
                [self.x, self.p],
                [self.f],
            )
            self.f_jac_func = casadi.Function(
                f"{model.__name__}_objective_jac",
                [self.x, self.p],
                [casadi.jacobian(self.f, self.x)],
            )

            if self.method is OptimizationProblem.Method.scipy_trust_constr:
                g_split = casadi.vertsplit(self.g)
                g_jacs = [casadi.jacobian(g, self.x) for g in g_split]

                scipy_constraints = []

                nonlinear_flags = [
                    casadi.depends_on(g_jac, self.x) for g_jac in g_jacs
                ]

                # process nonlinear constraints
                nonlinear_g_fun_exprs = [
                    g_expr for g_expr, is_nonlinear in zip(g_split, nonlinear_flags)
                    if is_nonlinear
                ]
                self.num_nonlinear_g = len(nonlinear_g_fun_exprs)
                if self.num_nonlinear_g:
                    nonlinear_g_jac_exprs = [
                        g_jac for g_jac, is_nonlinear in zip(g_jacs, nonlinear_flags)
                        if is_nonlinear
                    ]

                    self.g_func = casadi.Function(
                        f"{model.__name__}_nonlinear_constraint",
                        [self.x, self.p],
                        [casadi.vertcat(*nonlinear_g_fun_exprs)],
                    )


                    self.g_jac_func =  casadi.Function(
                        f"{model.__name__}_nonlinear_constraint_jac",
                        [self.x, self.p],
                        [casadi.vertcat(*nonlinear_g_jac_exprs)],
                    )


                    self.nonlinear_ub = np.array([
                        ub for ub, is_nonlinear in zip(self.ubg, nonlinear_flags)
                        if is_nonlinear
                    ])

                    self.nonlinear_lb = np.array([
                        lb for lb, is_nonlinear in zip(self.lbg, nonlinear_flags)
                        if is_nonlinear
                    ])


                # process linear constraints
                linear_A_exprs = [
                    g_jac for g_jac, is_nonlinear in zip(g_jacs, nonlinear_flags)
                    if not is_nonlinear
                ]
                self.num_linear_g = len(linear_A_exprs)
                if self.num_linear_g:

                    self.linear_jac_func = casadi.Function(
                        f"{model.__name__}_A",
                        [self.p], [casadi.vertcat(*linear_A_exprs)],
                    )

                    self.linear_ub = np.array([
                        ub for ub, is_nonlinear in zip(self.ubg, nonlinear_flags)
                        if not is_nonlinear
                    ])

                    self.linear_lb = np.array([
                        lb for lb, is_nonlinear in zip(self.lbg, nonlinear_flags)
                        if not is_nonlinear
                    ])









    def __call__(self, model_instance, *args):
        initializer_args = [self.x0]
        if self.has_p:
            p = casadi.vertcat(*flatten(args))
            initializer_args += [p]
        if not self.has_p or not isinstance(args[0], symbol_class):
            self.x0 = self.initializer_func(*initializer_args)
            if not isinstance(self.x0, casadi.DM):
                self.x0 = casadi.vertcat(
                    *self.x0
                )
            self.x0 = self.x0.toarray().reshape(-1)

        if self.method is OptimizationProblem.Method.ipopt:
            call_args = dict(
                x0=self.x0, ubx=self.ubx, lbx=self.lbx, ubg=self.ubg, lbg=self.lbg
            )

            out = self.optimizer(**call_args)
            if not self.has_p or not isinstance(args[0], symbol_class):
                self.x0 = out["x"]

            model_instance.bind_field(self.model.variable, out["x"])
            model_instance.bind_field(self.model.constraint, out["g"])
            model_instance.objective = out["f"].toarray()[0,0]
            self.stats = model_instance._stats = self.optimizer.stats()
        else:
            if self.has_p:
                extra_args = (p,)
            else:
                extra_args = ([],)

            scipy_constraints = []

            if self.method is OptimizationProblem.Method.scipy_cg:
                method_string = "CG"

            elif self.method is OptimizationProblem.Method.scipy_trust_constr:
                method_string = "trust-constr"
                if self.num_linear_g:
                    scipy_constraints.append(
                        LinearConstraint(
                            A = self.linear_jac_func(*extra_args).sparse(),
                            ub = self.linear_ub,
                            lb = self.linear_lb,
                            keep_feasible=True,
                        )
                    )
                if self.num_nonlinear_g:
                    scipy_constraints.append(
                        NonlinearConstraint(
                            fun=(lambda *args:
                                 self.g_func(*args, *extra_args).toarray().squeeze()
                            ),
                            jac=(lambda *args:
                                 self.g_jac_func(*args, *extra_args).sparse()
                            ),
                            lb=self.nonlinear_lb, ub=self.nonlinear_ub
                        )
                    )



            min_out = minimize(
                lambda *args: self.f_func(*args).toarray().squeeze(),
                self.x0,
                jac=lambda *args: self.f_jac_func(*args).toarray().squeeze(),
                method=method_string,
                args = extra_args,
                constraints = scipy_constraints,
            )
            model_instance.bind_field(self.model.variable, min_out.x)
            model_instance.objective = min_out.fun
            self.x0 = min_out.x
            self.stats = model_instance._stats = min_out

def get_state_setter(field, setter_args, default=0.):
    """
    field is MatchedField matching to the state
    default None to pass through
    """
    setter_exprs = []
    if isinstance(setter_args, symbol_class):
        setter_args = [setter_args]

    for state in field._matched_to:
        setter_symbol = field.get(match=state)
        if isinstance(setter_symbol, list) and len(setter_symbol) == 0:
            if default is not None:
                setter_symbol = default
            else:
                #setter_symbol = casadi.vertcat(*flatten(state.backend_repr)
                setter_symbol = (state.backend_repr).reshape((-1,1))
        elif isinstance(setter_symbol, co.BaseSymbol):
            setter_symbol = setter_symbol.backend_repr
        else:
            raise ValueError

        if isinstance(setter_symbol, symbol_class):
            setter_exprs.extend(
                casadi.vertsplit(setter_symbol.reshape((state.size,1)))
            )
        else:
            # symbol_class is always a matrix, and broadcast_to cannot handle (n,) to (n,1)
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
    def __init__(self, model, use_lam_te_p=True, include_lam_dot=False,
                 integrator_options={}):
        self.model = model
        self.ode_model = ode_model = model.inner_to

        self.use_lam_te_p = use_lam_te_p
        self.include_lam_dot = include_lam_dot
        self.integrator_options = integrator_options


        self.p = casadi.vertcat(*flatten(model.parameter))
        self.x = casadi.vertcat(*flatten(ode_model.state))
        self.lamda = symbol_class.sym("lambda", ode_model.state._count)
        self.simulation_signature = [
            self.p,
            self.ode_model.t,
            self.x,
        ]
        self.sym_event_channel = symbol_class.sym('event_channel')

        self.traj_out_expr = casadi.vertcat(*flatten(
            model.trajectory_output
        ))

        integrand_terms = flatten(model.trajectory_output.list_of('integrand'))
        self.traj_out_integrand = casadi.vertcat(*integrand_terms)
        self.traj_out_integrand_func = casadi.Function(
            f"{model.__name__}_trajectory_output_integrand",
            self.simulation_signature,
            [self.traj_out_integrand]
        )

        traj_out_names = model.trajectory_output.list_of('name')

        terminal_terms = flatten(model.trajectory_output.list_of('terminal_term'))
        self.traj_out_terminal_term = casadi.vertcat(*terminal_terms)
        self.traj_out_terminal_term_func = casadi.Function(
            f"{model.__name__}_trajectory_output_terminal_term",
            self.simulation_signature,
            [self.traj_out_terminal_term]
        )

        self.x0 = get_state_setter(model.initial, self.p,)
        dx0_dp_expr = casadi.jacobian(self.x0.expr, self.p)
        self.dx0_dp = casadi.Function(
            f"{model.__name__}_x0_jacobian",
            [self.p],
            [dx0_dp_expr],
        )
        self.dx0_dp.expr = dx0_dp_expr

        state_equation_func = get_state_setter(
            ode_model.dot,
            self.simulation_signature
        )
        if len(ode_model.output):
            self.y_expr = casadi.vertcat(*flatten(ode_model.output))
        else:
            self.y_expr = state_equation_func.expr
        self.e_exprs = [event.function for event in ode_model.Event.subclasses]
        self.h_exprs = [
            get_state_setter(
                event.update,
                self.simulation_signature,
                default=None,
            ).expr*(self.sym_event_channel==idx)
            if not getattr(event, 'terminate', False)

            else casadi.if_else(
                (self.sym_event_channel==idx),
                np.full((ode_model.state._count,), np.nan,),
                0.,
            )


            for idx, event in enumerate(ode_model.Event.subclasses)
        ]
        self.event_time_only = [
            casadi.depends_on(e.function, ode_model.t) and
            casadi.jacobian(e.function, ode_model.t).is_constant() and
            not sum([
                casadi.depends_on(e.function, state)
                for state in ode_model.state.list_of('backend_repr')
            ])


            for e in ode_model.Event.subclasses
        ]
        self.exact_times = [
            getattr(event, 'at_time', [casadi.inf])[0]
            for event in ode_model.Event.subclasses
        ]

        self.sim_shape_data = dict(
            dim_state = ode_model.state._count,
            dim_output = self.y_expr.shape[0],
            num_events = len(ode_model.Event.subclasses),
        )
        self.sim_func_kwargs = dict(
            state_equation_function = state_equation_func,
            output_equation_function = casadi.Function(
                f"{ode_model.__name__}_output",
                self.simulation_signature,
                [self.y_expr],
            ),
            event_equation_function = casadi.Function(
                f"{ode_model.__name__}_event",
                self.simulation_signature,
                [casadi.vertcat(*self.e_exprs)],
            ),
            update_equation_function = casadi.Function(
                f"{ode_model.__name__}_update",
                self.simulation_signature + [self.sym_event_channel],
                [sum(self.h_exprs)],
            ),

        )

        self.adjoint_signature = [
            self.p,
            self.x,
            self.ode_model.t,
            self.lamda,
        ]
        # TODO: is there something more pythonic than repeated, very similar list
        # comprehensions?
        self.lamda0s = [
            casadi.jacobian(terminal_term, self.x) for terminal_term in terminal_terms
        ]
        self.lamda0_funcs = [
            casadi.Function(
                f"{model.__name__}_{traj_name}_lamda0",
                self.adjoint_signature[:-1],
                [lamda0],
            ) for lamda0, traj_name in zip(self.lamda0s, traj_out_names)
        ]
        self.grad0s = [
            casadi.jacobian(terminal_term, self.p) for terminal_term in terminal_terms
        ]
        self.grad0_funcs = [
            casadi.Function(
                f"{model.__name__}_{traj_name}_grad0",
                self.adjoint_signature[:-1],
                [grad0],
            ) for grad0, traj_name in zip(self.grad0s, traj_out_names)
        ]

        lamda_jac = casadi.jacobian(state_equation_func.expr, self.x).T
        grad_jac = casadi.jacobian(state_equation_func.expr, self.p)

        self.lamda_dots = [
            -lamda_jac @ self.lamda - casadi.jacobian(integrand_term, self.x).T
            for integrand_term in integrand_terms
        ]
        self.grad_dots = [
            self.lamda.T @ grad_jac + casadi.jacobian(integrand_term, self.p)
            for integrand_term in integrand_terms
        ]

        self.lamda_dot_funcs = [
            casadi.Function(
                f"{model.__name__}_{traj_name}_lamda_dot",
                self.adjoint_signature,
                [lamda_dot],
            ) for lamda_dot, traj_name in zip(self.lamda_dots, traj_out_names)
        ]
        self.grad_dot_funcs = [
            casadi.Function(
                f"{model.__name__}_{traj_name}_grad_dot",
                self.adjoint_signature,
                [grad_dot],
            ) for grad_dot, traj_name in zip(self.grad_dots, traj_out_names)
        ]

        # lamda updates
        # grad updates
        self.dte_dxs = []
        self.d2te_dxdts = []
        self.dh_dxs = []

        self.dte_dps = []
        self.d2te_dpdts = []
        self.dh_dps = []

        for event, e_expr, h_expr in zip(
            ode_model.Event.subclasses, self.e_exprs, self.h_exprs
        ):
            dg_dx = casadi.jacobian(e_expr, self.x)
            dg_dt = casadi.jacobian(e_expr, ode_model.t)
            dg_dp = casadi.jacobian(e_expr, self.p)

            dte_dx = dg_dx/(dg_dx@state_equation_func.expr)
            dte_dp = dg_dp/(dg_dx@state_equation_func.expr + dg_dt)

            d2te_dxdt = (
                casadi.jacobian(dte_dx, ode_model.t)
                + casadi.jacobian(dte_dx, self.x) @ state_equation_func.expr
            )
            d2te_dpdt = (
                casadi.jacobian(dte_dp, ode_model.t)
                + casadi.jacobian(dte_dp, self.x) @ state_equation_func.expr
            )

            dh_dx = casadi.jacobian(h_expr, self.x)
            dh_dp = casadi.jacobian(h_expr, self.p)


            self.dte_dxs.append(
                casadi.Function(
                    f"{event.__name__}_dte_dx",
                    self.simulation_signature,
                    [dte_dx]
                )
            )
            self.dte_dxs[-1].expr = dte_dx

            self.dte_dps.append(
                casadi.Function(
                    f"{event.__name__}_dte_dp",
                    self.simulation_signature,
                    [dte_dp]
                )
            )
            self.dte_dps[-1].expr = dte_dp

            self.d2te_dxdts.append(
                casadi.Function(
                    f"{event.__name__}_d2te_dxdt",
                    self.simulation_signature,
                    [d2te_dxdt]
                )
            )
            self.d2te_dxdts[-1].expr = d2te_dxdt

            self.d2te_dpdts.append(
                casadi.Function(
                    f"{event.__name__}_d2te_dpdt",
                    self.simulation_signature,
                    [d2te_dpdt]
                )
            )
            self.d2te_dpdts[-1].expr = d2te_dpdt


            self.dh_dxs.append(
                casadi.Function(
                    f"{event.__name__}_dh_dx",
                    self.simulation_signature + [self.sym_event_channel],
                    [dh_dx]
                )
            )
            self.dh_dxs[-1].expr = dh_dx

            self.dh_dps.append(
                casadi.Function(
                    f"{event.__name__}_dh_dp",
                    self.simulation_signature + [self.sym_event_channel],
                    [dh_dp]
                )
            )
            self.dh_dps[-1].expr = dh_dp





        self.callback = ShootingGradientMethod(self)


    def __call__(self, model_instance, *args):
        out = self.callback(casadi.vertcat(*flatten(args)))
        if isinstance(out, casadi.MX):
            out = casadi.vertsplit(out)

        model_instance.bind_field(
            self.model.trajectory_output,
            out,
        )
        if hasattr(self.callback, 'res'):
            res = self.callback.res
            model_instance._res = res
            model_instance.t = res.t
            model_instance.bind_field(
                self.ode_model.state,
                res.x.T,
                wrap=True,
            )
            model_instance.bind_field(
                self.ode_model.output,
                res.y.T,
                wrap=True,
            )




