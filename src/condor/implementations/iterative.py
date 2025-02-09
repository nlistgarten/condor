from .utils import options_to_kwargs
from enum import Enum, auto
import casadi
import condor as co

from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize
from condor.backends.casadi.algebraic_solver import SolverWithWarmStart
from condor.backends.casadi.utils import (flatten, wrap)
from condor.solvers.casadi_warmstart_wrapper import (
    CasadiIterationCallback, CasadiNlpsolWarmstart
)
import numpy as np


from condor.backend import (
    symbol_class, callables_to_operator, expression_to_operator,
)



class InitializerMixin:
    def __init__(self, model):
        self.model = model

    def set_initial(self, *args, **kwargs):
        if args:
            # assume it's all initialized values, so flatten and assign
            self.x0 = flatten(args)
        else:
            count_accumulator = 0
            for field in self.parsed_initialized_fields:
                size_cum_sum = np.cumsum([count_accumulator] + field.list_of("size"))
                for start_idx, end_idx, name in zip(
                    size_cum_sum, size_cum_sum[1:], field.list_of("name")
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
                            solver_var.initializer.reshape((solver_var.size, 1))
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
                    if np.array(solver_var.warm_start).any():
                        x0_at_construction.extend(shaped_initial)
                        initializer_exprs.extend(
                            casadi.vertsplit(solver_var.backend_repr.reshape((-1, 1)))
                        )
                    else:
                        x0_at_construction.append(shaped_initial)
                        initializer_exprs.append(shaped_initial)

        self.parsed_initialized_fields = fields

        # setattr(self, f"{field._name}_at_construction", flatten(x0_at_construction))
        self.initial_at_construction = flatten(x0_at_construction)
        initializer_exprs = flatten(initializer_exprs)
        # setattr(self, f"{field._name}_initializer_func",
        self.initializer_func = casadi.Function(
            f"{field._model_name}_{field._name}_initializer",
            initializer_args,
            initializer_exprs,
        )


class AlgebraicSystem(InitializerMixin):
    def __init__(self, model_instance):
        model = model_instance.__class__
        model_instance.options_dict=options_to_kwargs(model)
        self.construct(model, **model_instance.options_dict)
        self(model_instance)

    def construct(
        self,
        model,
        # standard options
        atol=1e-12,
        rtol=1e-12,
        warm_start=True,
        exact_hessian=True,
        max_iter=100,
        # new options
        re_initialize=slice(
            None
        ),  # slice for re-initializing at every call. will support indexing
        default_initializer=0.0,
        error_on_fail=False,
    ):
        rootfinder_options = dict(
            error_on_fail=error_on_fail, abstol=atol, abstolStep=rtol, max_iter=max_iter
        )
        self.x = model.variable.flatten()
        self.g0 = model.residual.flatten()
        self.g1 = model.output.flatten()
        self.p = model.parameter.flatten()

        self.output_func = expression_to_operator(
            [self.x, self.p], self.g1, f"{model.__name__}_output"
        )

        self.model = model
        self.parse_initializers(
            model.variable,
            [self.x, self.p],
        )

        self.callback = SolverWithWarmStart(
            model.__name__,
            self.x,
            self.p,
            self.g0,
            self.g1,
            # use field dataclass to bind bounds?? then can flatten
            flatten(model.variable.list_of("lower_bound")),
            flatten(model.variable.list_of("upper_bound")),
            # self.implicit_output_at_construction,
            self.initial_at_construction,
            rootfinder_options,
            self.initializer_func,
            max_iter=max_iter,
        )

    def __call__(self, model_instance):
        self.call_args = model_instance.parameter.flatten()
        out = self.callback(self.call_args)
        self.var_out = self.model.variable.wrap(
            out[: self.model.variable._count]
        )
        model_instance.bind_field(self.var_out)
        model_instance.bind_field(
            self.model.output.wrap(
                self.output_func(self.var_out.flatten(), self.call_args)
            )
        )

        if hasattr(self.callback, "resid"):
            # TODO: verify this will never become stale
            # TODO: it seems a bit WET to wrap each field and then, eventually, bind
            # each field.
            resid = self.callback.resid
            model_instance.bind_field(
                self.model.residual.wrap(resid), symbols_to_instance=False
            )

        for k, v in model_instance.variable.asdict().items():
            model_var = getattr(self.model, k)
            if model_var.warm_start and not isinstance(
                model_var.initializer, symbol_class
            ):
                if not isinstance(v, symbol_class):
                    model_var.initializer = v

    def set_initial(self, *args, **kwargs):
        self.x0 = self.callback.x0
        super().set_initial(*args, **kwargs)
        self.callback.x0 = self.x0


class OptimizationProblem(InitializerMixin):
    # take an OptimizationProblem model with or without iteration spec and other Options
    # process options, create appropriate callback hooks
    # create appropriate operator --
    # "direct" nlpsol, nlpsol with warmstartwrapper, operator for scipy
    # technically even direct nlpsol should go through a set of functions so that it can
    # be used with a different backend
    # use expression_to_operator to create f, g, and if needed init_x0 and update_x0
    # hook to call other non-casadi optimizer using scipy infrastructure? cvxopt, cvxpy
    # was there a similar library to cvxopt?
    # maybe 3 different scipy implementations with a common base class
    # single casadi nlpsol implementation
    # 
    def make_warm_start(self, x0=None, lam_g0=None, lam_x0=None):
        if x0 is not None:
            self.x0 = x0
        if lam_g0 is not None:
            self.lam_g0 = lam_g0
        if lam_x0 is not None:
            self.lam_x0 = lam_x0


    def __init__(self, model_instance):
        model = model_instance.__class__
        model_instance.options_dict=options_to_kwargs(model)
        self.construct(model, **model_instance.options_dict)
        self(model_instance)

    default_options = {}

    def construct(
        self,
        model,
        iter_callback=None,
        init_callback=None,
        **options,
    ):
        self.model = model
        self.options = self.default_options.copy()
        self.options.update(options)

        self.iter_callback = iter_callback
        self.init_callback = init_callback

        self.f = f = getattr(model, "objective", 0)
        self.has_p = bool(len(model.parameter))
        self.p = p = model.parameter.flatten()
        self.x = x = model.variable.flatten()
        self.g = g = model.constraint.flatten()

        self.warm_start = model.variable.flatten("warm_start")
        self.initializer = model.variable.flatten("initializer")

        self.objective_func = expression_to_operator(
            [self.x, self.p],
            f,
            f"{model.__name__}_objective",
        )
        self.constraint_func = expression_to_operator(
            [self.x, self.p],
            g,
            f"{model.__name__}_constraint",
        )
        self.initializer_func = expression_to_operator(
            [self.p],
            self.initializer,
            f"{model.__name__}_initializer",
        )


        self.lbx = lbx = model.variable.flatten("lower_bound")
        self.ubx = ubx = model.variable.flatten("upper_bound")
        self.lbg = lbg = model.constraint.flatten("lower_bound")
        self.ubg = ubg = model.constraint.flatten("upper_bound")





    def load_initializer(self, model_instance):
        initializer_args = [self.x0]
        if self.has_p:
            self.eval_p = p = model_instance.parameter.flatten()
            initializer_args += [p]
        if not self.has_p or not isinstance(p, symbol_class):
            self.x0 = self.initializer_func(*initializer_args)
            if not isinstance(self.x0, casadi.DM):
                self.x0 = casadi.vertcat(*self.x0)
            self.x0 = self.x0.toarray().reshape(-1)

    def __call__(self, model_instance):
        self.load_initializer(model_instance)
        self.run_optimizer(model_instance)
        self.write_initializer(model_instance)

    def write_initializer(self, model_instance):
        for k, v in model_instance.variable.asdict().items():
            model_var = getattr(self.model, k)
            if np.array(model_var.warm_start).any() and not isinstance(
                model_var.initializer, symbol_class
            ) and not isinstance(v, symbol_class):
                model_var.initializer = v

class CasadiNlpsolImplementation(OptimizationProblem):
    class Method(Enum):
        ipopt = auto()
        snopt = auto()
        qrsqp = auto()

    method_strings = {
        Method.ipopt: "ipopt",
        Method.snopt: "snopt",
        Method.qrsqp: "sqpmethod",
    }


    method_default_options = {
        Method.ipopt: dict(
            #warm_start_init_point="no",
            warm_start_init_point="yes",
            sb="yes",  # suppress banner
        ),
        Method.snopt: dict(
            warm_start_init_point=False,
        ),
    }

    @property
    def default_options(self):
        return self.method_default_options.get(self.method, dict())

    def construct(
        self,
        model,
        exact_hessian=True,  # False -> ipopt alias for limited memory
        method=Method.ipopt,
        calc_lam_x=False,
        # ipopt specific, default = False may help with computing sensitivity. To do
        # proper warm start, need to provide lam_x and lam_g so I assume need calc_lam_x
        # = True
        **options,
    ):
        self.method = method
        super().construct(model, **options)

        self.nlp_args = dict(f=self.f, x=self.x, g=self.g)
        if co.backend.get_symbol_data(self.p).size > 1:
            self.nlp_args["p"] = self.p

        self.nlp_opts = dict()
        if self.iter_callback is not None:
            self.nlp_opts["iteration_callback"] = CasadiIterationCallback(
                "iter", self.nlp_args, model, self.iter_callback
            )

        self.method = method
        if self.method is CasadiNlpsolImplementation.Method.ipopt:
            self.nlp_opts.update(
                print_time=False,
                ipopt=self.options,
                # print_level = 0-2: nothing, 3-4: summary, 5: iter table (default)
                # tol=1E-14, # tighter tol for sensitivty
                # accept_every_trial_step="yes",
                # max_iter=1000,
                # constr_viol_tol=10.,
                bound_consistency=True,
                clip_inactive_lam=True,
                calc_lam_x=calc_lam_x,
                calc_lam_p=False,
            )
            # additional options from https://groups.google.com/g/casadi-users/c/OdRQKR13R50/m/bIbNoEHVBAAJ
            # to try to get sensitivity from ipopt. so far no...
            if not exact_hessian:
                self.nlp_opts["ipopt"].update(
                    hessian_approximation="limited-memory",
                )

        elif self.method is OptimizationProblem.Method.snopt:
            pass
        elif self.method is OptimizationProblem.Method.qrsqp:
            self.nlp_opts.update(
                qpsol="qrqp",
                qpsol_options=dict(
                    print_iter=False,
                    error_on_fail=False,
                ),
                # qpsol='osqp',
                verbose=False,
                tol_pr=1e-16,
                tol_du=1e-16,
                # print_iteration=False,
                print_time=False,
                # hessian_approximation= "limited-memory",
                print_status=False,
            )

        self.callback = CasadiNlpsolWarmstart(
            primary_function = self.objective_func,
            constraint_function = self.constraint_func,
            n_variable = model.variable._count,
            n_parameter = model.parameter._count,
            init_var = self.initializer_func,
            warm_start = self.warm_start,
            lbx = self.lbx,
            ubx = self.ubx,
            lbg = self.lbg,
            ubg = self.ubg,
            method_string = CasadiNlpsolImplementation.method_strings[method],
            options = self.nlp_opts,
            model_name = model.__name__,
        )

        self.optimizer = self.callback.optimizer
        self.nlp_args = self.callback.nlp_args
        self.lam_g0 = None
        self.lam_x0 = None

    def __call__(self, model_instance):
        self.run_optimizer(model_instance)
        self.write_initializer(model_instance)

    def run_optimizer(self, model_instance):
        if self.init_callback is not None:
            self.init_callback(model_instance.parameter, self.nlp_opts)

        run_p = model_instance.parameter.flatten()
        var_out = self.callback(run_p)

        if isinstance(var_out, symbol_class):
            out = dict(
                x = var_out,
                p = run_p,
                g = self.constraint_func(var_out, run_p),
                f = self.objective_func(var_out, run_p),
                lam_g = None,
                lam_x = None,
            )
        else:
            out = self.callback.out
            model_instance._stats = self.callback._stats

        model_instance.bind_field(self.model.variable.wrap(out["x"]))
        model_instance.bind_field(self.model.constraint.wrap(out["g"]))
        model_instance.objective = np.array(out["f"]).squeeze()
        self.out = out
        self.lam_g0 = out["lam_g"]
        self.lam_x0 = out["lam_x"]


class ScipyMinimizeBase(OptimizationProblem):
    def construct(
        self,
        model,
        **options,
    ):
        super().construct(model, **options)
        self.f_func = self.objective_func
        self.f_jac_func = casadi.Function(
            f"{model.__name__}_objective_jac",
            [self.x, self.p],
            [casadi.jacobian(self.f, self.x)],
        )
        self.g_split = casadi.vertsplit(self.g)
        initializer_args = [self.x]
        if self.has_p:
            initializer_args += [self.p]
        self.parse_initializers(model.variable, initializer_args)
        self.x0 = self.initial_at_construction.copy()

    def prepare_constraints(self, extra_args):
        return []

    def run_optimizer(self, model_instance):

            if self.has_p:
                extra_args = (self.eval_p,)
            else:
                extra_args = ([],)

            scipy_constraints = self.prepare_constraints(extra_args)

            if self.init_callback is not None:
                self.init_callback(
                    model_instance.parameter,
                    {**self.options, "lbx": self.lbx, "ubx": self.ubx},
                )

            min_out = minimize(
                lambda *args: self.f_func(*args).toarray().squeeze(),
                self.x0,
                jac=lambda *args: self.f_jac_func(*args).toarray().squeeze(),
                method=self.method_string,
                args=extra_args,
                constraints=scipy_constraints,
                bounds=np.vstack([self.lbx, self.ubx]).T,
                # tol = 1E-9,
                # options=dict(disp=True),
                options=self.options,
                callback=SciPyIterCallbackWrapper.create_or_none(
                    self.model,
                    model_instance.parameter,
                    self.iter_callback
                ),
            )

            model_instance.bind_field(self.model.variable.wrap(min_out.x))
            model_instance.objective = min_out.fun
            self.x0 = min_out.x
            self.stats = model_instance._stats = min_out

class ScipyCG(ScipyMinimizeBase):
    method_string = "CG"

class ScipySLSQP(ScipyMinimizeBase):
    method_string = "SLSQP"

    def construct(self, model, *args, **kwargs):
        super().construct(model, *args, **kwargs)
        self.equality_con_exprs = []
        self.inequality_con_exprs = []
        self.con = []
        for g, lbg, ubg in zip(self.g_split, self.lbg, self.ubg):
            if lbg == ubg:
                self.equality_con_exprs.append(g - lbg)
            else:
                if lbg > -np.inf:
                    self.inequality_con_exprs.append(g - lbg)
                if ubg < np.inf:
                    self.inequality_con_exprs.append(ubg - g)

        if self.equality_con_exprs:
            self.equality_con_expr = casadi.vertcat(*self.equality_con_exprs)
            self.eq_g_func = casadi.Function(
                f"{model.__name__}_equality_constraint",
                [self.x, self.p],
                [self.equality_con_expr],
            )
            self.eq_g_jac_func = casadi.Function(
                f"{model.__name__}_equality_constraint_jac",
                [self.x, self.p],
                [casadi.jacobian(self.equality_con_expr, self.x)],
            )
            self.con.append(
                dict(
                    type="eq",
                    fun=lambda x, p: self.eq_g_func(x, p).toarray().squeeze().T,
                    jac=self.eq_g_jac_func,
                )
            )

        if self.inequality_con_exprs:
            self.inequality_con_expr = casadi.vertcat(
                *self.inequality_con_exprs
            )
            self.ineq_g_func = casadi.Function(
                f"{model.__name__}_inequality_constraint",
                [self.x, self.p],
                [self.inequality_con_expr],
            )
            self.ineq_g_jac_func = casadi.Function(
                f"{model.__name__}_inequality_constraint_jac",
                [self.x, self.p],
                [casadi.jacobian(self.inequality_con_expr, self.x)],
            )
            self.con.append(
                dict(
                    type="ineq",
                    fun=lambda x, p: self.ineq_g_func(x, p)
                    .toarray()
                    .squeeze()
                    .T,
                    jac=self.ineq_g_jac_func,
                )
            )


    def prepare_constraints(self, extra_args):
        scipy_constraints = self.con
        for con in scipy_constraints:
            con["args"] = extra_args
        return scipy_constraints



class ScipyTrustConstr(ScipyMinimizeBase):
    method_string = "trust-constr"

    def construct(
        self,
        model,
        exact_hessian=True, # only trust-constr can use hessian at all
        keep_feasible=True,  # flag that goes to scipy trust constr linear constraints

        **options,
    ):
        super().construct(model, **options)
        self.keep_feasible = keep_feasible
        g_jacs = [casadi.jacobian(g, self.x) for g in g_split]

        scipy_constraints = []

        nonlinear_flags = [
            casadi.depends_on(g_jac, self.x)
            or casadi.depends_on(g_expr, self.p)
            for g_jac, g_expr in zip(g_jacs, g_split)
            # could ignore dependence on parameters, but to be consistent with
            # ipopt specification require parameters within function not in
            # bounds
        ]

        # process nonlinear constraints
        nonlinear_g_fun_exprs = [
            g_expr
            for g_expr, is_nonlinear in zip(g_split, nonlinear_flags)
            if is_nonlinear
        ]
        self.num_nonlinear_g = len(nonlinear_g_fun_exprs)
        if self.num_nonlinear_g:
            nonlinear_g_jac_exprs = [
                g_jac
                for g_jac, is_nonlinear in zip(g_jacs, nonlinear_flags)
                if is_nonlinear
            ]

            self.g_func = casadi.Function(
                f"{model.__name__}_nonlinear_constraint",
                [self.x, self.p],
                [casadi.vertcat(*nonlinear_g_fun_exprs)],
            )

            self.g_jac_func = casadi.Function(
                f"{model.__name__}_nonlinear_constraint_jac",
                [self.x, self.p],
                [casadi.vertcat(*nonlinear_g_jac_exprs)],
            )

            self.nonlinear_ub = np.array(
                [
                    ub
                    for ub, is_nonlinear in zip(self.ubg, nonlinear_flags)
                    if is_nonlinear
                ]
            )

            self.nonlinear_lb = np.array(
                [
                    lb
                    for lb, is_nonlinear in zip(self.lbg, nonlinear_flags)
                    if is_nonlinear
                ]
            )

        # process linear constraints
        linear_A_exprs = [
            g_jac
            for g_jac, is_nonlinear in zip(g_jacs, nonlinear_flags)
            if not is_nonlinear
        ]
        self.num_linear_g = len(linear_A_exprs)
        if self.num_linear_g:
            self.linear_jac_func = casadi.Function(
                f"{model.__name__}_A",
                [self.p],
                [casadi.vertcat(*linear_A_exprs)],
            )

            self.linear_ub = np.array(
                [
                    ub
                    for ub, is_nonlinear in zip(self.ubg, nonlinear_flags)
                    if not is_nonlinear
                ]
            )

            self.linear_lb = np.array(
                [
                    lb
                    for lb, is_nonlinear in zip(self.lbg, nonlinear_flags)
                    if not is_nonlinear
                ]
            )

    def prepare_constraints(self, extra_args):
        scipy_constraints = []
        if self.num_linear_g:
            scipy_constraints.append(
                LinearConstraint(
                    A=self.linear_jac_func(*extra_args).sparse(),
                    ub=self.linear_ub,
                    lb=self.linear_lb,
                    keep_feasible=self.keep_feasible,
                )
            )
        if self.num_nonlinear_g:
            scipy_constraints.append(
                NonlinearConstraint(
                    fun=(
                        lambda *args: self.g_func(*args, *extra_args)
                        .toarray()
                        .squeeze()
                    ),
                    jac=(
                        lambda *args: self.g_jac_func(
                            *args, *extra_args
                        ).sparse()
                    ),
                    lb=self.nonlinear_lb,
                    ub=self.nonlinear_ub,
                )
            )
        return scipy_constraints


import inspect
class SciPyIterCallbackWrapper:
    @classmethod
    def create_or_none(cls, model, parameters, callback=None):
        if callback is None:
            return None
        return cls(model, parameters, callback)

    def __init__(self, model, parameters, callback):
        self.iter = 0
        self.callback = callback
        self.model = model
        self.parameters = parameters
        self.pass_instance = len(inspect.signature(self.callback).parameters) > 4

    def __call__(self, xk, res=None):
        variable = self.model.variable.wrap(xk)
        instance = self.model.from_values(
            **self.parameters.asdict(),
            **variable.asdict()
        )
        callback_args = (
            self.iter, variable, instance.objective, instance.constraint,
        )
        if self.pass_instance:
            callback_args += (instance,)

        self.callback(*callback_args)
        self.iter += 1

