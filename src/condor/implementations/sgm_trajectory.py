from .utils import options_to_kwargs
from enum import Enum, auto
import condor as co
import condor.solvers.sweeping_gradient_method as sgm
import numpy as np
from condor import backend
import casadi

flatten = co.backend.utils.flatten
vertcat = co.backend.utils.vertcat
substitute = co.backend.utils.substitute
recurse_if_else = co.backend.utils.recurse_if_else
symbol_class = co.backend.utils.symbol_class

FunctionsToOperator = co.backend.utils.FunctionsToOperator

def get_state_setter(field, setter_args, setter_targets=None, default=0.0, subs={}):
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

    target_sizes = [target.size for target in setter_targets]
    start_indexes = np.cumsum([0] + target_sizes)

    setter_exprs = symbol_class(sum(target_sizes), 1)
    for state, start_index, end_index in zip(
        setter_targets, start_indexes, start_indexes[1:]
    ):
        setter_element = field.get(match=state)
        if isinstance(setter_element, list) and len(setter_element) == 0:
            if default is not None:
                setter_exprs[start_index:end_index] = default
            else:
                # setter_element = casadi.vertcat(*flatten(state.backend_repr)
                setter_exprs[start_index:end_index] = (state.backend_repr).reshape(
                    (-1, 1)
                )
        elif isinstance(setter_element, co.BaseElement):
            setter_exprs[start_index:end_index] = setter_element.backend_repr.reshape(
                (-1, 1)
            )
        else:
            raise ValueError

    setter_exprs = substitute(setter_exprs, subs)
    setter_exprs = [setter_exprs]
    setter_func = casadi.Function(
        f"{field._model_name}_{field._matched_to._name}_{field._name}",
        setter_args,
        setter_exprs,
        dict(
            allow_free=True,
        ),
    )
    setter_func.expr = setter_exprs[0]
    if setter_func.expr.shape != setter_args[-1].shape and len(setter_args) > 1:
        raise ValueError("This must be a bug... a shape mismatch")
    return setter_func


class TrajectoryAnalysis:
    class Solver(Enum):
        CVODE = auto()
        dopri5 = auto()
        dop853 = auto()

    def __init__(self, model_instance, args):
        model = model_instance.__class__
        self.construct(model, **options_to_kwargs(model))
        self(model_instance, *args)

    def construct(
        self,
        model,
        state_atol=1e-12,
        state_rtol=1e-6,
        state_adaptive_max_step_size=0.0,
        state_max_step_size=0,
        # TODO add options for scipy solver name (dopri5 or dop853) and settings for the
        # rootfinder, including choosing between brentq and newton
        adjoint_atol=1e-12,
        adjoint_rtol=1e-6,
        adjoint_adaptive_max_step_size=4.0,
        adjoint_max_step_size=0,
        state_solver=Solver.dopri5,
        adjoint_solver=Solver.dopri5,
        # lmm_type=ADAMS or BDF, possibly also linsolver, etc? for CVODE?
        # other options for scipy.ode + event rootfinder?
    ):
        self.model = model
        self.ode_model = ode_model = model._meta.primary

        self.x = casadi.vertcat(*flatten(model.state))
        self.lamda = symbol_class.sym("lambda", model.state._count)

        self.p = casadi.vertcat(*flatten(model.parameter))

        self.simulation_signature = [
            self.p,
            self.model.t,
            self.x,
        ]

        self.traj_out_expr = casadi.vertcat(*flatten(model.trajectory_output))
        self.can_sgm = isinstance(self.p, casadi.MX) and isinstance(
            self.traj_out_expr, casadi.MX
        )

        traj_out_names = model.trajectory_output.list_of("name")

        integrand_terms = flatten(model.trajectory_output.list_of("integrand"))
        self.traj_out_integrand = casadi.vertcat(*integrand_terms)
        traj_out_integrand_func = casadi.Function(
            f"{model.__name__}_trajectory_output_integrand",
            self.simulation_signature,
            [self.traj_out_integrand],
            dict(
                allow_free=True,
            ),
        )

        terminal_terms = flatten(model.trajectory_output.list_of("terminal_term"))
        self.traj_out_terminal_term = casadi.vertcat(*terminal_terms)
        traj_out_terminal_term_func = casadi.Function(
            f"{model.__name__}_trajectory_output_terminal_term",
            self.simulation_signature,
            [self.traj_out_terminal_term],
            dict(
                allow_free=True,
            ),
        )

        self.state0 = get_state_setter(
            model.initial,
            self.p,
        )

        control_subs_pairs = {
            control.backend_repr: [(control.default,)] for control in ode_model.modal
        }
        for mode in model._meta.modes:
            for act in mode.action:
                control_subs_pairs[act.match.backend_repr].append(
                    (mode.condition, act.backend_repr)
                )
        control_sub_expression = {}
        for k, v in control_subs_pairs.items():
            control_sub_expression[k] = substitute(
                recurse_if_else(v), control_sub_expression
            )

        state_equation_func = get_state_setter(
            model.dot,
            self.simulation_signature,
            subs=control_sub_expression,
        )
        lamda_jac = casadi.jacobian(state_equation_func.expr, self.x).T
        state_dot_jac_func = casadi.Function(
            f"{ode_model.__name__}_state_jacobian",
            self.simulation_signature,
            [lamda_jac.T],
            dict(
                allow_free=True,
            ),
        )

        self.e_exprs = []
        self.h_exprs = []
        at_time_slices = [
            sgm.NextTimeFromSlice(
                casadi.Function(
                    f"{ode_model.__name__}_at_times_t0",
                    [self.p],
                    # TODO in future allow t0 to occur at arbitrary times
                    [0.0, 0.0, casadi.inf],
                )
            )
        ]

        terminating = []

        events = [e for e in model._meta.events]
        if model.tf is not np.inf:

            class Terminate(ode_model.Event):
                at_time = (model.tf,)
                terminate = True

            events += [Terminate]
            ode_model.Event._meta.subclasses = ode_model.Event._meta.subclasses[:-1]

        num_events = len(events)

        for event_idx, event in enumerate(events):
            if (event.function is not np.nan) == (event.at_time is not np.nan):
                raise ValueError(
                    f"Event class `{event}` has set both `function` and `at_time`"
                )
            if getattr(event, "function", np.nan) is not np.nan:
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

                    e_expr = (
                        at_time.step
                        * casadi.sin(
                            casadi.pi * (model.t - at_time_start) / at_time.step
                        )
                        / (casadi.pi * 100)
                    )
                    # self.events(solver_res.values.t, solver_res.values.y, gs)
                    # e_expr = casadi.sin(casadi.pi*(ode_model.t-at_time_start)/at_time.step)
                    e_expr = casadi.fmod(model.t - at_time_start, at_time.step)

                    # TODO: verify start and stop for at_time slice
                    if isinstance(at_time_start, symbol_class) or at_time_start != 0.0:
                        e_expr = e_expr * (model.t >= at_time_start)
                        # if there is a start offset, add a linear term to provide a
                        # zero-crossing at first occurance
                        pre_term = (at_time_start - model.t) * (
                            model.t <= at_time_start
                        )
                    else:
                        pre_term = 0

                    if at_time.stop is not None:
                        e_expr = e_expr * (model.t <= at_time.stop)
                        # if there is an end-time, hold constant to prevent additional
                        # zero crossings -- hopefully works even if stop is on an event
                        # post_term = (ode_model.t >= at_time.stop) * at_time.step*casadi.sin(casadi.pi*(at_time.stop-at_time_start)/at_time.step)/casasadi.pi
                        post_term = (model.t >= at_time.stop) * casadi.fmod(
                            at_time.stop - at_time_start, at_time.step
                        )
                        at_time_stop = at_time.stop
                    else:
                        post_term = 0
                        at_time_stop = casadi.inf

                    e_expr = e_expr + pre_term + post_term

                    at_time_slices.append(
                        sgm.NextTimeFromSlice(
                            casadi.Function(
                                f"{ode_model.__name__}_at_times_{event_idx}",
                                [self.p],
                                [at_time_start, at_time_stop, at_time.step],
                            )
                        )
                    )
                else:
                    if isinstance(at_time[0], co.BaseElement):
                        at_time0 = at_time[0].backend_repr
                    else:
                        at_time0 = at_time[0]
                    e_expr = at_time0 - model.t
                    at_time_slices.append(
                        sgm.NextTimeFromSlice(
                            casadi.Function(
                                f"{ode_model.__name__}_at_times_{event_idx}",
                                [self.p],
                                [at_time0, at_time0, casadi.inf],
                            )
                        )
                    )

            self.e_exprs.append(e_expr)

            if event.terminate:
                # For simupy, use nans to trigger termination; do we ever want to allow
                # an update to a terminating event?
                # self.h_exprs.append(
                #    casadi.MX.nan(ode_model.state._count)
                # )
                terminating.append(event_idx)

            h_expr = get_state_setter(
                event.update,
                self.simulation_signature,
                setter_targets=[state for state in model.state],
                default=None,
                subs=control_sub_expression,
            )
            self.h_exprs.append(h_expr)

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

        # self.e_exprs = substitute(casadi.vertcat(*self.e_exprs), control_sub_expression)

        if len(model.dynamic_output):
            # self.y_expr = casadi.vertcat(*flatten(ode_model.dynamic_output))
            self.y_expr = casadi.vertcat(*flatten(model.dynamic_output))
            self.y_expr = substitute(self.y_expr, control_sub_expression)
            self.dynamic_output_func = casadi.Function(
                f"{ode_model.__name__}_dynamic_output",
                self.simulation_signature,
                [self.y_expr],
            )
        else:
            self.dynamic_output_func = None

        self.state_system = sgm.System(
            dim_state=model.state._count,
            initial_state=self.state0,
            dot=state_equation_func,
            jac=state_dot_jac_func,
            time_generator=sgm.TimeGeneratorFromSlices(at_time_slices),
            events=casadi.Function(
                f"{ode_model.__name__}_event",
                self.simulation_signature,
                [substitute(casadi.vertcat(*self.e_exprs), control_sub_expression)],
                dict(
                    allow_free=True,
                ),
            ),
            updates=self.h_exprs,
            num_events=num_events,
            terminating=terminating,
            dynamic_output=self.dynamic_output_func,
            atol=state_atol,
            rtol=state_rtol,
            adaptive_max_step=state_adaptive_max_step_size,
            max_step_size=state_max_step_size,
            solver_class=state_solver_class,
        )
        self.at_time_slices = at_time_slices



        self.p_state0_p_p_expr = casadi.jacobian(self.state0.expr, self.p)

        p_state0_p_p = casadi.Function(
            f"{model.__name__}_x0_jacobian",
            [self.p],
            [self.p_state0_p_p_expr],
            dict(
                allow_free=True,
            ),
        )
        p_state0_p_p.expr = self.p_state0_p_p_expr

        self.adjoint_signature = [
            self.p,
            self.x,
            self.model.t,
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
            )
            for lamdaF, traj_name in zip(self.lamdaFs, traj_out_names)
        ]
        self.gradFs = [
            casadi.jacobian(terminal_term, self.p) for terminal_term in terminal_terms
        ]
        self.gradF_funcs = [
            casadi.Function(
                f"{model.__name__}_{traj_name}_gradF",
                self.simulation_signature,
                [gradF],
            )
            for gradF, traj_name in zip(self.gradFs, traj_out_names)
        ]

        grad_jac = casadi.jacobian(state_equation_func.expr, self.p)

        param_dot_jac_func = casadi.Function(
            f"{model.__name__}_param_jacobian",
            self.simulation_signature,
            [grad_jac],
            dict(
                allow_free=True,
            ),
        )

        state_integrand_jacs = [
            casadi.jacobian(integrand_term, self.x).T
            for integrand_term in integrand_terms
        ]
        state_integrand_jac_funcs = [
            casadi.Function(
                f"{model.__name__}_state_integrand_jac_{ijac_expr_idx}",
                self.simulation_signature,
                [ijac_expr],
                dict(
                    allow_free=True,
                ),
            )
            for ijac_expr_idx, ijac_expr in enumerate(state_integrand_jacs)
        ]

        param_integrand_jacs = [
            casadi.jacobian(integrand_term, self.p).T
            for integrand_term in integrand_terms
        ]
        param_integrand_jac_funcs = [
            casadi.Function(
                f"{model.__name__}_param_integrand_jac_{ijac_expr_idx}",
                self.simulation_signature,
                [ijac_expr],
            )
            for ijac_expr_idx, ijac_expr in enumerate(param_integrand_jacs)
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

        for event, e_expr, h_expr in zip(events, self.e_exprs, self.h_exprs):
            dg_dx = casadi.jacobian(e_expr, self.x)
            dg_dt = casadi.jacobian(e_expr, model.t)
            dg_dp = casadi.jacobian(e_expr, self.p)

            dte_dx = dg_dx / (dg_dx @ state_equation_func.expr)
            dte_dp = -dg_dp / (dg_dx @ state_equation_func.expr + dg_dt)

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
                    f"{event.__name__}_dte_dp", self.simulation_signature, [dte_dp]
                )
            )
            self.dte_dps[-1].expr = dte_dp

            dh_dx = substitute(dh_dx, control_sub_expression)
            self.dh_dxs.append(
                casadi.Function(
                    f"{event.__name__}_dh_dx", self.simulation_signature, [dh_dx]
                )
            )
            self.dh_dxs[-1].expr = dh_dx

            dh_dp = substitute(dh_dp, control_sub_expression)
            self.dh_dps.append(
                casadi.Function(
                    f"{event.__name__}_dh_dp", self.simulation_signature, [dh_dp]
                )
            )
            self.dh_dps[-1].expr = dh_dp

        self.trajectory_analysis_sgm = sgm.TrajectoryAnalysisSGM(
            state_system = self.state_system,
            integrand_terms=traj_out_integrand_func,
            terminal_terms=traj_out_terminal_term_func,

            dte_dxs=self.dte_dxs,
            dh_dxs=self.dh_dxs,
            state_jac=state_dot_jac_func,

            adjoint_atol=adjoint_atol,
            adjoint_rtol=adjoint_rtol,
            adjoint_adaptive_max_step_size=adjoint_adaptive_max_step_size,
            adjoint_max_step_size=adjoint_max_step_size,
            adjoint_solver_class=adjoint_solver_class,

            p_x0_p_params=p_state0_p_p,
            p_dots_p_params=param_dot_jac_func,
            dh_dps=self.dh_dps,
            dte_dps=self.dte_dps,
            p_terminal_terms_p_params=self.gradF_funcs,
            p_integrand_terms_p_params=param_integrand_jac_funcs,
            p_terminal_terms_p_state=self.lamdaF_funcs,
            p_integrand_terms_p_state=state_integrand_jac_funcs,
        )

        wrapper_funcs = [
            self.trajectory_analysis_sgm.function,
            self.trajectory_analysis_sgm.jacobian,
        ]

        self.callback = FunctionsToOperator(
            wrapper_funcs, self, jacobian_of=None,
            input_symbol = self.p,
            output_symbol = self.traj_out_expr,
        )
        self.callback.construct()

        if not self.can_sgm:
            return


    def __call__(self, model_instance, *args):
        self.callback.from_implementation = True
        self.args = casadi.vertcat(*flatten(args))
        self.out = self.callback(self.args)
        self.callback.from_implementation = False

        if hasattr(self.trajectory_analysis_sgm, "res"):
            res = self.trajectory_analysis_sgm.res
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





