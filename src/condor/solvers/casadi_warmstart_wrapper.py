import casadi
from dataclasses import dataclass
import numpy as np
import condor.backends.casadi as backend


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
    init_x0: list # initial guess function of p,  x0(p)
    warm_start: list

    options = None

    def __post_init__(self):
        super().__post_init__()
        if self.update_x0 is None:
            self.update_x0 = lambda x, p: self.init_x0(p)


@dataclass
class CasadiNlpsolWarmstart(CasadiWarmstartWrapperBase):
    """
    nlpsol
    """

    constraint = None # optional constraint function g(x, p)
    lbg = None
    # bound for constraint function so lbg <= g(x, p); required if constraint function
    # is provided, not used if not
    ubg = None
    # bound for constraint function so g(x, p) <= ubg; required if constraint function
    # is provided, not used if not

    def __post_init__(self):
        super().__post_init__()
        self.objective = self.primary_function

    @property
    def function(self):
        pass

    @property
    def jacobian(self):
        # compute casadi jacobian object thing, and return it
        # if overallbackend is casadi, will re-enter casadi native for additional
        # derivatives
        return None

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
