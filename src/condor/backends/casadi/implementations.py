import casadi
import condor as co
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from condor.backends.casadi.utils import (
    flatten, wrap, symbol_class, substitute, recurse_if_else
)
from condor.backends.casadi.algebraic_solver import SolverWithWarmStart
import condor.solvers.shooting_gradient_method as sgm
import condor.backends.casadi.shooting_gradient_method as ca_sgm
from condor.backends.casadi.table_lookup import NDSplinesCallback

from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

# TODO: make SGM and SolverWithWarmStart (really, back-tracking solver and possibly only
# needed if broyden doesn't resolve it?) generic and figure out how to separate the
# algorithm from the casadi callback. I think SWWS already does this pretty well

# Implementations should definitely be the owners of expressions used to generate
# functions -- maybe own all intermediates used to interface with solver? I think some
# casadi built-in solvers+symbolic representation take either (or both) expressions and
# ca.Function`s (~ duck-typed functions that can operate on either symbols or numerics,
# kind of an arechetype of callback?) but even when they take expressions, I don't think
# they allow access to them after creation so it's useful for the implementation to keep
# it

# would like implementation to be the interface between model field and the callback,
# but some of the helpers (eg initializer, state settter, etc) generate functions, not
# just collect expressions.

# TODO figure out how to use names for casadi callback layer
# TODO function generation is primarily responsibility of callback (e.g., nlpsol takes
# expressions, not functions)
# TODO if we provide a backend reference to vertcat, implementations can be
# backend-agnostic and just provide appropriate flattening + binding of fields! may
# also/instead create callback even for OptimizationProblem, ExplicitSystem to provide
# consistent interface
# --> move some of the helper functions that are tightly coupled to backend to utils, ?
# and generalize, eg state setter
# TODO for custom solvers like SGM, table, does the get_jacobian arguments allow you to 
# avoid computing wrt particular inputs/outputs if possible?


class DeferredSystem:
    def __init__(self, model):
        symbol_inputs = model.input.list_of("backend_repr")
        symbol_outputs = model.output.list_of("backend_repr")

        self.symbol_inputs = [casadi.vertcat(*flatten(symbol_inputs))]
        self.symbol_outputs = [casadi.vertcat(*flatten(symbol_outputs))]

        name_inputs = model.input.list_of('name')
        name_outputs = model.output.list_of('name')
        self.model = model
        self.func =  casadi.Function(
            model.__name__, symbol_inputs, symbol_outputs, dict(allow_free=True,),
        )

    def __call__(self, model_instance, *args):
        model_instance.bind_field(
            self.model.output, 
            self.symbol_outputs[0],
        )

class ExplicitSystem:
    def __init__(self, model):
        symbol_inputs = model.input.list_of("backend_repr")
        symbol_outputs = model.output.list_of("backend_repr")

        self.symbol_inputs = [casadi.vertcat(*flatten(symbol_inputs))]
        self.symbol_outputs = [casadi.vertcat(*flatten(symbol_outputs))]

        name_inputs = model.input.list_of('name')
        name_outputs = model.output.list_of('name')
        self.model = model
        #self.func =  casadi.Function(model.__name__, self.symbol_inputs, self.symbol_outputs)
        self.func =  casadi.Function(
            model.__name__, self.symbol_inputs, self.symbol_outputs, dict(allow_free=True,),
        )

    def __call__(self, model_instance, *args):
        self.args = casadi.vertcat(*flatten(args))
        self.out = self.func(self.args)
        model_instance.bind_field(
            self.model.output, 
            self.out,
        )

class Tablelookup:
    def __init__(self, model, degrees=3,):
        self.model = model
        self.symbol_inputs = model.input.list_of("backend_repr")
        self.symbol_outputs = model.output.list_of("backend_repr")
        self.input_data = [model.input_data.get(match=inp).backend_repr for inp in model.input]
        self.output_data = np.stack(
            [model.output_data.get(match=out).backend_repr for out in model.output],
            axis=-1,
        )
        self.degrees = degrees
        self.callback = NDSplinesCallback(self)

    def __call__(self, model_instance, *args):
        out = self.callback(casadi.vertcat(*flatten(args)))
        if isinstance(out, casadi.MX):
            out = casadi.vertsplit(out)
        model_instance.bind_field(
            self.model.output,
            out,
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
        error_on_fail=False,
        enforce_bounds=False,
    ):
        rootfinder_options = dict(
            error_on_fail=error_on_fail,
            abstol=atol,
            abstolStep=rtol,
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
            enforce_bounds=enforce_bounds,
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
        snopt = auto()
        qrsqp = auto()
        scipy_cg = auto()
        scipy_trust_constr = auto()
        scipy_slsqp = auto()

    scipy_trust_constr_option_defaults = dict(xtol=1E-8,)

    def make_warm_start(self, x0=None, lam_g0=None, lam_x0=None):
        if x0 is not None:
            self.x0 = x0
        if lam_g0 is not None:
            self.lam_g0 = lam_g0
        if lam_x0 is not None:
            self.lam_x0 = lam_x0

    default_options = {
        Method.ipopt: dict(
            warm_start_init_point=False,
        ),
    }


    def __init__(
        self, model,
        exact_hessian=True, # False -> ipopt alias for limited memory
        keep_feasible = True, # flag that goes to scipy trust constr linear constraints
        method=Method.ipopt,
        calc_lam_x = False, 
        # ipopt specific, default = False may help with computing sensitivity. To do
        # proper warm start, need to provide lam_x and lam_g so I assume need calc_lam_x
        # = True
        **options,
    ):
        self.model = model
        self.options = self.default_options.get(method, dict()).copy()
        self.options.update(options)

        self.keep_feasible = keep_feasible


        self.f = f = getattr(model, 'objective', 0)
        self.has_p = bool(len(model.parameter))
        self.p = p = casadi.vertcat(*flatten(model.parameter))
        self.x = x = casadi.vertcat(*flatten(model.variable))
        self.g = g = casadi.vertcat(*flatten(model.constraint))


        self.objective_func = casadi.Function(
            f"{model.__name__}_objective",
            [self.x, self.p],
            [self.f],
        )
        self.constraint_func = casadi.Function(
            f"{model.__name__}_constraint",
            [self.x, self.p],
            [g],
        )

        self.lbx = lbx = casadi.vcat(flatten(model.variable.list_of("lower_bound"))).toarray().reshape(-1)
        self.ubx = ubx = casadi.vcat(flatten(model.variable.list_of("upper_bound"))).toarray().reshape(-1)
        self.lbg = lbg = casadi.vcat(flatten(model.constraint.list_of("lower_bound"))).toarray().reshape(-1)
        self.ubg = ubg = casadi.vcat(flatten(model.constraint.list_of("upper_bound"))).toarray().reshape(-1)

        self.nlp_args = dict(f=f, x=x, g=g)
        initializer_args = [self.x]
        if self.has_p:
            initializer_args += [self.p]
            self.nlp_args["p"] = p
        self.parse_initializers(model.variable, initializer_args)

        self.x0 = self.initial_at_construction.copy()
        self.lam_g0 = None
        self.lam_x0 = None
        #self.variable_at_construction.copy()

        self.method = method
        if self.method is OptimizationProblem.Method.ipopt:
            self.ipopt_opts = ipopt_opts = dict(
                print_time= False,
                ipopt = options,
                # print_level = 0-2: nothing, 3-4: summary, 5: iter table (default)
                #tol=1E-14, # tighter tol for sensitivty
                #accept_every_trial_step="yes",
                #max_iter=1000,
                #constr_viol_tol=10.,

                bound_consistency=True,
                clip_inactive_lam=True,
                calc_lam_x=calc_lam_x,
                calc_lam_p=False,
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
        elif self.method is OptimizationProblem.Method.snopt:
            self.optimizer = casadi.nlpsol(
                model.__name__,
                "snopt",
                self.nlp_args,
                #self.ipopt_opts,
            )
        elif self.method is OptimizationProblem.Method.qrsqp:

            self.qrsqp_opts = dict(
                qpsol='qrqp',
                qpsol_options=dict(
                    print_iter=False,
                    error_on_fail=False,
                ),

                #qpsol='osqp',

                verbose=False,
                tol_pr=1E-16,
                tol_du=1E-16,
                #print_iteration=False,
                print_time=False,
                #hessian_approximation= "limited-memory",
                print_status=False,
            )

            self.optimizer = casadi.nlpsol(
                model.__name__,
                'sqpmethod',
                self.nlp_args,
                self.qrsqp_opts
            )

        else:
            self.optimizer = None
            self.f_func = self.objective_func
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
                    casadi.depends_on(g_jac, self.x) or casadi.depends_on(g_expr, self.p)
                    for g_jac, g_expr in zip(g_jacs, g_split)
                    # could ignore dependence on parameters, but to be consistent with
                    # ipopt specification require parameters within function not in
                    # bounds
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

        if self.method in (
            OptimizationProblem.Method.ipopt,
            OptimizationProblem.Method.snopt,
            OptimizationProblem.Method.qrsqp,
        ):
            call_args = dict(
                x0=self.x0, ubx=self.ubx, lbx=self.lbx, ubg=self.ubg, lbg=self.lbg, 
            )
            if self.options["warm_start_init_point"]:
                if self.lam_x0 is not None:
                    call_args.update(lam_x0= self.lam_x0)
                if self.lam_g0 is not None:
                    call_args.update(lam_g0= self.lam_g0)

            if self.has_p:
                call_args["p"] = p

            out = self.optimizer(**call_args)
            if not self.has_p or not isinstance(args[0], symbol_class):
                self.x0 = out["x"]

            model_instance.bind_field(self.model.variable, out["x"])
            model_instance.bind_field(self.model.constraint, out["g"])
            model_instance.objective = np.array(out["f"]).squeeze()
            self.stats = model_instance._stats = self.optimizer.stats()
            self.out = model_instance._out = out
            self.lam_g0 = out["lam_g"]
            self.lam_x0 = out["lam_x"]
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
                            keep_feasible=self.keep_feasible,
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
                #options=dict(disp=True),
                options=self.options,
            )
            model_instance.bind_field(self.model.variable, min_out.x)
            model_instance.objective = min_out.fun
            self.x0 = min_out.x
            self.stats = model_instance._stats = min_out

def get_state_setter(field, setter_args, setter_targets=None, default=0., subs={}):
    """
    used for building functions from matched fields to their match (loops over "state"
    but is generalized for any matched fields)
    used for dot, update, initial condition,

    field is MatchedField
    default None to pass through
    setter_args are arguments for generalized 

    """
    setter_exprs = []
    if not isinstance(setter_args, list):
        setter_args = [setter_args]

    if setter_targets is None:
        setter_targets = [state for state in field._matched_to]
    for state in setter_targets:
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
            setter_symbol = np.array(setter_symbol)
            if (state.size ==1 and setter_symbol.size ==1) or state.shape == setter_symbol.shape:
                setter_exprs.append(
                    setter_symbol.reshape((state.size,1))
                )
            elif state.shape[1] == 1:
                setter_exprs.append(
                    np.broadcast_to(setter_symbol, state.size).reshape(-1)
                )
            else:
                setter_exprs.append(
                    np.broadcast_to(setter_symbol, state.shape).reshape(-1)
                )
    #setter_exprs = flatten(setter_exprs)
    setter_exprs = casadi.vertcat(*setter_exprs)
    setter_exprs = substitute(setter_exprs, subs)
    setter_exprs = [setter_exprs]
    setter_func = casadi.Function(
        f"{field._model_name}_{field._matched_to._name}_{field._name}",
        setter_args,
        setter_exprs,
        dict(allow_free=True,),
    )
    setter_func.expr = setter_exprs[0]
    return setter_func


class TrajectoryAnalysis:

    class Solver(Enum):
        CVODE = auto()
        dopri5 = auto()
        dop853 = auto()

    def __init__(
        self, model,

        state_atol=1E-12, state_rtol=1E-6,
        state_adaptive_max_step_size=0., state_max_step_size = 0,
        # TODO add options for scipy solver name (dopri5 or dop853) and settings for the
        # rootfinder, including choosing between brentq and newton

        adjoint_atol=1E-12, adjoint_rtol=1E-6,
        adjoint_adaptive_max_step_size=4., adjoint_max_step_size = 0,

        state_solver = Solver.dopri5,
        adjoint_solver = Solver.dopri5,

        #lmm_type=ADAMS or BDF, possibly also linsolver, etc? for CVODE?
        # other options for scipy.ode + event rootfinder?
    ):
        self.model = model
        self.ode_model = ode_model = model.inner_to

        self.x = casadi.vertcat(*flatten(model.state))
        self.lamda = symbol_class.sym("lambda", model.state._count)

        self.p = casadi.vertcat(*flatten(model.parameter))

        self.simulation_signature = [
            self.p,
            self.ode_model.t,
            self.x,
        ]

        self.traj_out_expr = casadi.vertcat(*flatten(
            model.trajectory_output
        ))
        self.can_sgm = isinstance(self.p, casadi.MX) and isinstance(self.traj_out_expr, casadi.MX)

        traj_out_names = model.trajectory_output.list_of('name')

        integrand_terms = flatten(model.trajectory_output.list_of('integrand'))
        self.traj_out_integrand = casadi.vertcat(*integrand_terms)
        traj_out_integrand_func = casadi.Function(
            f"{model.__name__}_trajectory_output_integrand",
            self.simulation_signature,
            [self.traj_out_integrand],
            dict(allow_free=True,),
        )

        terminal_terms = flatten(model.trajectory_output.list_of('terminal_term'))
        self.traj_out_terminal_term = casadi.vertcat(*terminal_terms)
        traj_out_terminal_term_func = casadi.Function(
            f"{model.__name__}_trajectory_output_terminal_term",
            self.simulation_signature,
            [self.traj_out_terminal_term],
            dict(allow_free=True,),
        )


        self.state0 = get_state_setter(model.initial, self.p,)

        control_subs_pairs = {
            control.backend_repr: [(control.default,)]
            for control in ode_model.modal
        }
        for mode in ode_model.Mode._meta.subclasses:
            for act in mode.action:
                control_subs_pairs[act.match.backend_repr].append(
                    (mode.condition, act.backend_repr)
                )
        control_sub_expression = {}
        for k,v in control_subs_pairs.items():
            control_sub_expression[k] = substitute(
                recurse_if_else(v), control_sub_expression
            )

        state_equation_func = get_state_setter(
            ode_model.dot,
            self.simulation_signature,
            subs = control_sub_expression,
        )
        lamda_jac = casadi.jacobian(state_equation_func.expr, self.x).T
        state_dot_jac_func = casadi.Function(
            f"{ode_model.__name__}_state_jacobian",
            self.simulation_signature,
            [lamda_jac.T],
            dict(allow_free=True,),
        )

        self.e_exprs = []
        self.h_exprs = []
        at_time_slices = [sgm.NextTimeFromSlice(
            casadi.Function(
                f"{ode_model.__name__}_at_times_t0",
                [self.p],
                # TODO in future allow t0 to occur at arbitrary times
                [0., 0., casadi.inf]
            )
        )]

        terminating = []


        model_tf = getattr(model, 'tf', None)
        if model_tf is not None:
            class Terminate(ode_model.Event):
                at_time = model_tf,
                terminate = True

        events = [ e for e in ode_model.Event._meta.subclasses ]

        if model_tf is not None:
            ode_model.Event._meta.subclasses = ode_model.Event._meta.subclasses[:-1]

        num_events = len(events)

        for event_idx, event in enumerate(events):
            terminate = getattr(event, 'terminate', False)
            if (
                getattr(event, 'function', None) is not None
            ) == (
                getattr(event, 'at_time', None) is not None
            ):
                raise ValueError(
                    f"Event class `{event}` has set both `function` and `at_time`"
                )
            if getattr(event, 'function', None) is not None:
                e_expr = event.function
            else:
                at_time = event.at_time
                if isinstance(at_time, slice):
                    if at_time.step is None:
                        raise ValueError

                    if at_time.start is not None:
                        at_time_start = at_time.start
                    else:
                        at_time_start = 0

                    e_expr = at_time.step*casadi.sin(casadi.pi*(ode_model.t-at_time_start)/at_time.step)/(
                        casadi.pi*100
                    )
                    # self.events(solver_res.values.t, solver_res.values.y, gs)
                    #e_expr = casadi.sin(casadi.pi*(ode_model.t-at_time_start)/at_time.step)
                    e_expr = casadi.fmod(ode_model.t - at_time_start, at_time.step) 

                    # TODO: verify start and stop for at_time slice
                    if isinstance(at_time_start, symbol_class) or at_time_start != 0.:
                        e_expr = e_expr * (ode_model.t >= at_time_start)
                        # if there is a start offset, add a linear term to provide a
                        # zero-crossing at first occurance
                        pre_term = (at_time_start - ode_model.t) * (ode_model.t <= at_time_start)
                    else:
                        pre_term = 0

                    if at_time.stop is not None:
                        e_expr = e_expr * (ode_model.t <= at_time.stop)
                        # if there is an end-time, hold constant to prevent additional
                        # zero crossings -- hopefully works even if stop is on an event
                        #post_term = (ode_model.t >= at_time.stop) * at_time.step*casadi.sin(casadi.pi*(at_time.stop-at_time_start)/at_time.step)/casasadi.pi
                        post_term = (ode_model.t >= at_time.stop) * casadi.fmod(at_time.stop - at_time_start, at_time.step)
                        at_time_stop = at_time.stop
                    else:
                        post_term = 0
                        at_time_stop = casadi.inf

                    e_expr = e_expr + pre_term + post_term

                    at_time_slices.append(sgm.NextTimeFromSlice(
                        casadi.Function(
                            f"{ode_model.__name__}_at_times_{event_idx}",
                            [self.p],
                            [at_time_start, at_time_stop, at_time.step]
                        )
                    ))
                else:
                    if isinstance(at_time[0], co.BaseSymbol):
                        at_time0 = at_time[0].backend_repr
                    else:
                        at_time0 = at_time[0]
                    e_expr = at_time0 - ode_model.t
                    at_time_slices.append(sgm.NextTimeFromSlice(
                        casadi.Function(
                            f"{ode_model.__name__}_at_times_{event_idx}",
                            [self.p],
                            [at_time0, at_time0, casadi.inf]
                        )
                    ))

            self.e_exprs.append(
                e_expr
            )

            if terminate:
                # For simupy, use nans to trigger termination; do we ever want to allow
                # an update to a terminating event?
                #self.h_exprs.append(
                #    casadi.MX.nan(ode_model.state._count)
                #)
                terminating.append(event_idx)
            self.h_exprs.append(
                get_state_setter(
                    event.update,
                    self.simulation_signature,
                    default=None,
                    subs = control_sub_expression,
                )
            )


        set_solvers = []
        for solver in [state_solver, adjoint_solver]:
            if solver is TrajectoryAnalysis.Solver.CVODE:
                solver_class = sgm.SolverCVODE
            elif solver is TrajectoryAnalysis.Solver.dopri5:
                solver_class = sgm.SolverSciPyDopri5
            elif solver is TrajectoryAnalysis.Solver.dop853:
                solver_class = sgm.SolverSciPyDop853
            set_solvers.append(solver_class)
        state_solver_class, adjoint_solver_class = set_solvers


        self.trajectory_analysis = sgm.TrajectoryAnalysis(
            traj_out_integrand_func, traj_out_terminal_term_func,
        )
        #self.e_exprs = substitute(casadi.vertcat(*self.e_exprs), control_sub_expression)

        if len(ode_model.dynamic_output):
            #self.y_expr = casadi.vertcat(*flatten(ode_model.dynamic_output))
            self.y_expr = casadi.vertcat(*flatten(model.dynamic_output))
            self.y_expr = substitute(self.y_expr, control_sub_expression)
            self.dynamic_output_func = casadi.Function(
                f"{ode_model.__name__}_dynamic_output",
                self.simulation_signature,
                [self.y_expr],
            )
        else:
            self.dynamic_output_func = None

        self.StateSystem = sgm.System(
            dim_state = model.state._count,
            initial_state = self.state0,
            dot = state_equation_func,
            jac = state_dot_jac_func,
            time_generator = sgm.TimeGeneratorFromSlices(
                at_time_slices
            ),
            events = casadi.Function(
                f"{ode_model.__name__}_event",
                self.simulation_signature,
                [substitute(casadi.vertcat(*self.e_exprs), control_sub_expression)],
                dict(allow_free=True,),
            ),
            updates = self.h_exprs,
            num_events = num_events,
            terminating = terminating,
            dynamic_output = self.dynamic_output_func,
            atol=state_atol,
            rtol=state_rtol,
            adaptive_max_step=state_adaptive_max_step_size,
            max_step_size=state_max_step_size,
            solver_class = state_solver_class,
        )

        self.callback = ca_sgm.ShootingGradientMethod(self)

        if not self.can_sgm:
            return

        self.p_state0_p_p_expr = casadi.jacobian(self.state0.expr, self.p)

        p_state0_p_p = casadi.Function(
            f"{model.__name__}_x0_jacobian",
            [self.p],
            [self.p_state0_p_p_expr],
            dict(allow_free=True,),
        )
        p_state0_p_p.expr = self.p_state0_p_p_expr

        self.adjoint_signature = [
            self.p,
            self.x,
            self.ode_model.t,
            self.lamda,
        ]
        # TODO: is there something more pythonic than repeated, very similar list
        # comprehensions?
        self.lamdaFs = [
            casadi.jacobian(terminal_term, self.x) for terminal_term in terminal_terms
        ]
        self.lamdaF_funcs = [
            casadi.Function(
                f"{model.__name__}_{traj_name}_lamdaF",
                self.simulation_signature,
                [lamdaF],
            ) for lamdaF, traj_name in zip(self.lamdaFs, traj_out_names)
        ]
        self.gradFs = [
            casadi.jacobian(terminal_term, self.p) for terminal_term in terminal_terms
        ]
        self.gradF_funcs = [
            casadi.Function(
                f"{model.__name__}_{traj_name}_gradF",
                self.simulation_signature,
                [gradF],
            ) for gradF, traj_name in zip(self.gradFs, traj_out_names)
        ]

        grad_jac = casadi.jacobian(state_equation_func.expr, self.p)

        param_dot_jac_func = casadi.Function(
            f"{ode_model.__name__}_param_jacobian",
            self.simulation_signature,
            [grad_jac],
            dict(allow_free=True,),
        )

        state_integrand_jacs = [
            casadi.jacobian(integrand_term, self.x).T
            for integrand_term in integrand_terms
        ]
        state_integrand_jac_funcs = [
                casadi.Function(
                    f"{ode_model.__name__}_state_integrand_jac_{ijac_expr_idx}",
                    self.simulation_signature,
                    [ijac_expr],
                    dict(allow_free=True,),
                ) for ijac_expr_idx, ijac_expr in enumerate(state_integrand_jacs)
        ]

        param_integrand_jacs = [
            casadi.jacobian(integrand_term, self.p).T
            for integrand_term in integrand_terms
        ]
        param_integrand_jac_funcs = [
                casadi.Function(
                    f"{ode_model.__name__}_param_integrand_jac_{ijac_expr_idx}",
                    self.simulation_signature,
                    [ijac_expr]
                ) for ijac_expr_idx, ijac_expr in enumerate(param_integrand_jacs)
        ]
        # TODO figure out how to combine (and possibly reverse direction) to reduce
        # number of calls, since this is potentially most expensive call with inner
        # loop solvers, 

        # lamda updates
        # grad updates
        self.dte_dxs = []
        self.dh_dxs = []

        self.dte_dps = []
        self.dh_dps = []

        for event, e_expr, h_expr in zip(
            events, self.e_exprs, self.h_exprs
        ):
            dg_dx = casadi.jacobian(e_expr, self.x)
            dg_dt = casadi.jacobian(e_expr, ode_model.t)
            dg_dp = casadi.jacobian(e_expr, self.p)

            dte_dx = dg_dx/(dg_dx@state_equation_func.expr)
            dte_dp = -dg_dp/(dg_dx@state_equation_func.expr + dg_dt)


            dh_dx = casadi.jacobian(h_expr.expr, self.x)
            dh_dp = casadi.jacobian(h_expr.expr, self.p)


            """
            te = ode_model.t

            xtem = self.x
            xtep = h_expr(self.p, te, xtem)

            ftem = state_equation_func(self.p, te, xtem)
            ftep = state_equation_func(self.p, te, xtep)
            delta_fs = ftep - ftem
            delta_xs = xtep - xtem

            lamda_tep = self.lamda
            eyen = casadi.MX.eye(lamda_tep.shape[0])
            lamda_tem = (( eyen + dte_dx.T @ ftem.T) @  dh_dx.T - dte_dx.T @ ftep.T )@ lamda_tep

            delta_lamdas = (lamda_tem - lamda_tep)

            # TODO update for forcing function
            lamda_dot_tem = -state_dot_jac_func(self.p, te, xtem).T @ lamda_tem
            lamda_dot_tep = -state_dot_jac_func(self.p, te, xtep).T @ lamda_tep

            delta_lamda_dots = (lamda_dot_tem - lamda_dot_tep)

            jac_update = (
                lamda_tep.T @ dh_dp
                - lamda_tep.T @ ( ftep - dh_dx @ ftem ) @ dte_dp 
            )

            jac_update = substitute(jac_update, control_sub_expression)

            self.jac_updates.append(
                casadi.Function(
                    f"{event.__name__}_jac_update",
                    self.adjoint_signature,
                    [jac_update],
                )
            )
            """

            dte_dx = substitute(dte_dx, control_sub_expression)
            self.dte_dxs.append(
                casadi.Function(
                    f"{event.__name__}_dte_dx",
                    self.simulation_signature,
                    [dte_dx],
                )
            )
            self.dte_dxs[-1].expr = dte_dx


            dte_dp = substitute(dte_dp, control_sub_expression)
            self.dte_dps.append(
                casadi.Function(
                    f"{event.__name__}_dte_dp",
                    self.simulation_signature,
                    [dte_dp]
                )
            )
            self.dte_dps[-1].expr = dte_dp


            dh_dx = substitute(dh_dx, control_sub_expression)
            self.dh_dxs.append(
                casadi.Function(
                    f"{event.__name__}_dh_dx",
                    self.simulation_signature,
                    [dh_dx]
                )
            )
            self.dh_dxs[-1].expr = dh_dx

            dh_dp = substitute(dh_dp, control_sub_expression)
            self.dh_dps.append(
                casadi.Function(
                    f"{event.__name__}_dh_dp",
                    self.simulation_signature,
                    [dh_dp]
                )
            )
            self.dh_dps[-1].expr = dh_dp

        self.AdjointSystem = sgm.AdjointSystem(
            state_jac = state_dot_jac_func,
            dte_dxs = self.dte_dxs,
            dh_dxs = self.dh_dxs,
            atol=adjoint_atol,
            rtol=adjoint_rtol,
            adaptive_max_step=adjoint_adaptive_max_step_size,
            max_step_size=adjoint_max_step_size,
            solver_class = adjoint_solver_class,
        )
        self.shooting_gradient_method = sgm.ShootingGradientMethod(
            adjoint_system = self.AdjointSystem,
            p_x0_p_params = p_state0_p_p,
            p_dots_p_params = param_dot_jac_func,
            dh_dps = self.dh_dps,
            dte_dps = self.dte_dps,

            p_terminal_terms_p_params = self.gradF_funcs,
            p_integrand_terms_p_params = param_integrand_jac_funcs,
            p_terminal_terms_p_state = self.lamdaF_funcs,
            p_integrand_terms_p_state = state_integrand_jac_funcs,
        )


    def __call__(self, model_instance, *args):
        self.callback.from_implementation = True
        self.args = casadi.vertcat(*flatten(args))
        self.out = self.callback(self.args)
        self.callback.from_implementation = False

        if hasattr(self.callback, 'res'):
            res = self.callback.res
            model_instance._res = res
            model_instance.t = np.array(res.t)
            model_instance.bind_field(
                self.model.state,
                np.array(res.x).T,
                wrap=True,
            )
            if self.dynamic_output_func:
                yy = np.empty((model_instance.t.size, self.model.dynamic_output._count))
                for idx, (t, x) in enumerate(zip(res.t, res.x)):
                    yy[idx, None] = self.dynamic_output_func(res.p, t, x).T
                model_instance.bind_field(
                    self.model.dynamic_output,
                    yy.T,
                    wrap=True,
                )

        model_instance.bind_field(
            self.model.trajectory_output,
            self.out,
        )

