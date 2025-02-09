import casadi
from dataclasses import dataclass
import numpy as np
from condor.backends.casadi import CasadiFunctionCallback, symbol_class


@dataclass
class CasadiWarmstartWrapperBase:
    """
    baseclass to wrap Casadi nlpsol and rootfinder with a warmstart

    """
    primary_function: callable # callable f(x, p). nlpsol's objective or rootfinder's residual



    n_variable: int
    n_parameter: int

    lbx: list # lower bounds for variable
    ubx: list # upper bounds for variable
    init_var: list # initial guess function of p,  x0(p)
    warm_start: list
    method_string: str

    model_name: str = ""
    options: dict = None


    # I guess technically should provide jacobians and hessian wrt variable to construct
    # the casadi.callables_to_operator... perfectly sufficient for top-level/single call
    # but gets a bit clunky with parameter (e.g., embedded models that we want to take
    # derivative of), and the callables_to_operator currently assumes SISO.

    # I like having all the direct casadi stuff outside of implementaitons.iterative,
    # so not having casadi-speicifc constructions, but maybe it's OK for this case since
    # the solver is casadi itself?
    # basically because of the API requirements of casadi, either condor needs to better
    # define its MIMO (or maybe still MISO?) calculus API so we can go from
    # {condor.backend} operator to casadi for NLPSOL or we allow a bit of tighter
    # coupling.

    # for now, assume I'm getting backend-agnostic MISO operator (which for now means
    # casadi.Function of x and p) and we'll just take for granted we can construct a
    # full casadi expression for objective and constraints.

    # and even more confusing/inconsistent, casadi rootfinder takes MIMO function that
    # x,p input and residual,explicit outptus.

    # TODO

    def __post_init__(self):
        #super().__post_init__()
        #if self.update_x0 is None:
        #    self.update_x0 = lambda x, p: self.init_x0(p)
        self.any_warm_start = np.any(self.warm_start)
        self.all_warm_start = np.all(self.warm_start)
        if self.options is None:
            self.options = {}

    def has_jacobian(self):
        return self.jacobian is not None

    def get_jacobian(self, name, inames, onames, opts):
        return self.jacobian

    def get_sparsity_in(self, i):
        return self.placeholder_func.sparsity_in(i)

    def get_sparsity_out(self, i):
        return casadi.Sparsity.dense(*self.placeholder_func.sparsity_out(i).shape)


@dataclass
class CasadiNlpsolWarmstart(CasadiWarmstartWrapperBase, casadi.Callback):
    """
    nlpsol
    """

    constraint_function: ... = None # optional constraint function g(x, p)
    lbg: ... = None
    # bound for constraint function so lbg <= g(x, p); required if constraint function
    # is provided, not used if not
    ubg: ... = None
    # bound for constraint function so g(x, p) <= ubg; required if constraint function
    # is provided, not used if not

    def __post_init__(self):
        super().__post_init__()
        self.objective = self.primary_function

        self.x = x = symbol_class.sym(f"{self.model_name}_variable", (self.n_variable,1))
        self.p = p = symbol_class.sym(f"{self.model_name}_parameter", (self.n_parameter,1))

        f = self.primary_function(x,p)
        g = self.constraint_function(x,p)

        self.nlp_args = dict(f=f, x=x, g=g)
        if self.n_parameter:
            self.nlp_args["p"] = self.p

        self.lam_g0 = None
        self.lam_x0 = None
        self.last_x = None
        # self.variable_at_construction.copy()
        self.apply_lamda_initial = self.options.get(
            "ipopt", {}
        ).get("warm_start_init_point", "no") == "yes"

        self.optimizer = casadi.nlpsol(
            f"{self.model_name}_optimizer",
            self.method_string,
            self.nlp_args,
            self.options,
        )
        if not self.init_var.jacobian()(self.p, []).nnz():
            sym_x0 = self.init_var(np.zeros(self.p.shape))
        self.sym_opt_out = self.optimizer(
            x0=sym_x0,
            p=p,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
        )


        self.placeholder_func = casadi.Function(
            f"{self.model_name}_func",
            [p],
            [self.optimizer(
                x0=x,
                p=p,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
            )["x"]],
            dict(allow_free=True),
        )
        self.placeholder_func_xp = casadi.Function(
            f"{self.model_name}_func",
            [x,p],
            [self.optimizer(
                x0=x,
                p=p,
                lbx=self.lbx,
                ubx=self.lbx,
                lbg=self.lbg,
                ubg=self.ubg,
            )["x"]],
        )


        self.placeholder_func = casadi.Function(
            f"{self.model_name}_func",
            [p],
            [self.sym_opt_out["x"]],
        )
        try:
            self.jacobian = self.placeholder_func.jacobian()
        except RuntimeError as e:
            self.jacobian = None


        #self.jacobian = casadi.jacobian(self.placeholder_func_xp(x,p), p)


        casadi.Callback.__init__(self)
        self.construct(f"{self.model_name}_Callback", {})
        self.last_p = None

    def eval(self, args):
        run_p = args[0]

        run_kwargs = dict(
            ubx=self.ubx,
            lbx=self.lbx,
            ubg=self.ubg,
            lbg=self.lbg,
        )

        if self.n_parameter:
            run_kwargs.update(p=run_p)

        x0 = self.init_var(run_p)
        if self.any_warm_start and self.last_x is not None:
            for idx, warm_start in enumerate(self.warm_start):
                if warm_start:
                    x0[idx] = self.last_x[idx]
        if self.any_warm_start and self.last_x is not None:
            #breakpoint()
            pass
        run_kwargs.update(x0=x0)
        if self.all_warm_start and self.apply_lamda_initial:
            if self.lam_x0 is not None:
                run_kwargs.update(
                    lam_x0=self.lam_x0,
                )
            if self.lam_g0 is not None:
                run_kwargs.update(
                    lam_g0=self.lam_g0,
                )

        if np.any(run_kwargs["ubx"] <= run_kwargs["lbx"]):
            breakpoint()
        if self.last_p == run_p:
            print("repeat call", run_p)
            out = self.out
        else:
            print("new call running with", run_p)
            out = self.out = self.optimizer(**run_kwargs)
            self._stats = self.optimizer.stats()

        self.last_p = run_p

        self.last_x = out["x"]
        self.lam_g0 = out["lam_g"]
        self.lam_x0 = out["lam_x"]

        return out["x"],



class CasadiRootfinderWarmstart(CasadiWarmstartWrapperBase):
    def __post_init__(self):
        super().__post_init__()
        self.residual = self.primary_function



class CasadiIterationCallback(casadi.Callback):
    def __init__(self, name, nlpdict, model, iteration_callback, opts={}):
        casadi.Callback.__init__(self)
        self.iteration_callback = iteration_callback
        self.nlpdict = nlpdict
        self.model = model
        self.iter = 0
        self.construct(name, opts)

    def get_n_in(self):
        return casadi.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        n = casadi.nlpsol_out(i)
        return n

    def get_name_out(self, i):
        return "ret"

    def get_sparsity_in(self, i):
        n = casadi.nlpsol_out(i)
        if n == "f":
            return casadi.Sparsity.dense(1)
        elif n in ("x", "lam_x"):
            return self.nlpdict["x"].sparsity()
        elif n in ("g", "lam_g"):
            g = self.nlpdict["g"]
            if not hasattr(g, "sparsity"):
                return casadi.Sparsity.dense(
                    np.atleast_2d(g).shape
                )
            return g.sparsity()
        return casadi.Sparsity(0, 0)

    def eval(self, args):
        # x, f, g, lam_x, lam_g, lam_p
        x, f, g, *_ = args
        var_data = self.model.variable.wrap(x)
        constraint_data = self.model.constraint.wrap(g)
        self.iteration_callback(self.iter, var_data, f, constraint_data)
        self.iter += 1
        return [0]
