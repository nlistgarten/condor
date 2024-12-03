import numpy as np
from scipy import linalg
import casadi
from condor.backends.casadi.utils import CasadiFunctionCallbackMixin
from condor.solvers.newton import Newton

import logging

log = logging.getLogger(__name__)


class SolverWithWarmStart(CasadiFunctionCallbackMixin, casadi.Callback):
    def __init__(
        self,
        name,
        x,
        p,
        g0,
        g1,
        lbx,
        ubx,
        x0,
        rootfinder_options,
        initializer,
        enforce_bounds=True,
        max_iter=100,
    ):
        casadi.Callback.__init__(self)
        self.name = name
        self.initializer = initializer
        self.x0 = x0
        self.lbx = np.array(lbx).reshape(-1)
        self.ubx = np.array(ubx).reshape(-1)
        self.enforce_bounds = enforce_bounds

        self.newton = Newton(
            x,
            p,
            g0,
            lbx=lbx,
            ubx=ubx,
            tol=rootfinder_options["abstol"],
            error_on_fail=rootfinder_options["error_on_fail"],
            ls_type=None,
            max_iter=max_iter,
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
            [p],
            casadi.vertsplit(out_imp) + casadi.vertsplit(out_exp),
        )

        self.construct(name, {})

    def eval(self, args):
        p = casadi.vertcat(*args)
        log.debug(f"solver {self.name} with x0={self.x0}")
        self.x0 = self.initializer(self.x0, p)

        self.x0 = np.array(self.x0).reshape(-1)
        p = p.toarray().reshape(-1)

        lbx_violations = np.where(self.x0 < self.lbx)[0]
        ubx_violations = np.where(self.x0 > self.ubx)[0]
        self.x0[lbx_violations] = self.lbx[lbx_violations]
        self.x0[ubx_violations] = self.ubx[ubx_violations]

        if self.enforce_bounds:
            self.x0 = self.newton(self.x0, p)
            self.resid, out_exp = self.resid_func(self.x0, p)
            out_exp = out_exp.toarray().reshape(-1)
        else:
            self.x0, out_exp = self.rootfinder(self.x0, p)
            self.x0 = self.x0.toarray().reshape(-1)
            out_exp = out_exp.toarray().reshape(-1)
        log.debug(f"solved to {self.x0}")

        return tuple([*self.x0, *out_exp])
