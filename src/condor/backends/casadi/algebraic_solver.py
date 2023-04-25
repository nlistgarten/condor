import numpy as np
from scipy import linalg
import casadi
from condor.backends.casadi.utils import CasadiFunctionCallbackMixin

def wrap_ls_func(f):

    def func(x, p):
        out = f(x, p).toarray().reshape(-1)
        if out.size == 1:
            out = out[0]
        return out

    return func


class Newton:
    """
    ::

        x = casadi.vertcat(*[casadi.MX.sym(f"x{i}", 1) for i in range(4)])
        p = casadi.vertcat(*[casadi.MX.sym(f"p{i}", 1) for i in range(2)])
        resid = casadi.vertcat(x[0] + p[0], x[1] * p[1], x[2]**2, x[3] / 2)
        Newton(x, p, resids)

    ::

        f = casadi.Function("f", [x, p], [resid])
        fprime = casadi.Function("fprime", [x, p], [casadi.jacobian(resid, x)])

    """

    def __init__(
        self, x, p, resids, lbx=None, ubx=None, max_iter=1000, tol=1e-10, ls_type=None
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.ls_type = ls_type

        self.lbx = lbx
        if lbx is None:
            self.lbx = np.full(x.size()[0], -np.inf)
        self.ubx = ubx
        if ubx is None:
            self.ubx = np.full(x.size()[0], np.inf)

        # residuals for newton
        self.f = casadi.Function("f", [x, p], [resids])
        self.fprime = casadi.Function("fprime", [x, p], [casadi.jacobian(resids, x)])

        # obj for linesearch, TODO: performance testing for sumsqr vs norm_2
        #resids_norm = casadi.sumsqr(resids)
        #print("using sumsqr")
        resids_norm = casadi.norm_2(resids)
        print("using norm2")
        self.ls_f = wrap_ls_func(casadi.Function("ls_f", [x, p], [resids_norm]))
        self.ls_fprime = wrap_ls_func(
            casadi.Function("ls_fprime", [x, p], [casadi.jacobian(resids_norm, x)])
        )

    def __call__(self, x, p):
        x = np.asarray(x, dtype=float).reshape(-1)
        p = np.asarray(p, dtype=float).reshape(-1)

        itr = 0
        while True:
            if itr >= self.max_iter:
                print("newton failed, max iters reached")
                break

            # eval residuals func
            f = self.f(x, p).toarray().reshape(-1)

            # check tol
            if np.linalg.norm(f, ord=np.inf) < self.tol:
                break

            # eval jacobian
            J = self.fprime(x, p)

            # newton step calculation J @ d = -f
            dx = linalg.solve(J, -f)

            # TODO check stall

            if self.ls_type is None:
                x += dx
            elif "AG" in self.ls_type:
                # om sequence:
                #   init: given current x, full newton step dx
                #      take the step x += dx
                #      enforce bounds, modify step
                #   line search from bound-enforced point back to x?

                # TODO ArmijoGoldsteinLS takes an option for initial alpha, default 1.0

                xb, dxb = _enforce_bounds_scalar(x+dx, dx, 1.0, self.lbx, self.ubx)

                fx = self.ls_f(x, p)
                gx = self.ls_fprime(x, p)

                alpha, fc, gc, fout, _, _ = line_search(
                    self.ls_f,
                    self.ls_fprime,
                    x,
                    dxb,
                    gfk=gx,
                    old_fval=fx,
                    c1=0.1,
                    c2=0.9,
                    args=(p,),
                    maxiter=2,
                )

                # alpha, fc, fout = line_search_armijo(
                #     self.ls_f, x, dxb, gx, fx, args=(p,), c1=0.1, alpha0=1
                # )
                print("line search step:", alpha)

                if alpha is None:
                    # TODO: performance tuning, alpha = 0 [x = x], alpha = 1 [x=xb],
                    # alpha=0.5, or something else? 0 and 1 work for simple turbojet
                    # design cycle only
                    alpha = 0.5
                    print("line search failed!", xb)
                    print("setting alpha =", alpha)
                    break

                x += alpha * dxb
            else:
                x += dx
                x, dx = _enforce_bounds_scalar(x, dx, 1.0, self.lbx, self.ubx)

            itr += 1

        # TODO
        # self.resid_vals = f
        self.iters = itr
        return x


def _enforce_bounds_scalar(u, du, alpha, lower_bounds, upper_bounds):
    # from openmdao/solvers/linesearch/backtracking.py

    # The assumption is that alpha * step has been added to this vector
    # just prior to this method being called. We are currently in the
    # initialization of a line search, and we're trying to ensure that
    # the initial step does not violate bounds. If it does, we modify
    # the step vector directly.

    # If u > lower, we're just adding zero. Otherwise, we're adding
    # the step required to get up to the lower bound.
    # For du, we normalize by alpha since du eventually gets
    # multiplied by alpha.
    change_lower = 0. if lower_bounds is None else np.maximum(u, lower_bounds) - u

    # If u < upper, we're just adding zero. Otherwise, we're adding
    # the step required to get down to the upper bound, but normalized
    # by alpha since du eventually gets multiplied by alpha.
    change_upper = 0. if upper_bounds is None else np.minimum(u, upper_bounds) - u

    if lower_bounds is not None:
        mask = u < lower_bounds
        if np.any(mask):
            print("vals exceeding lower bounds")
            print("\tval:", u[mask])
            print("\tlower:", lower_bounds[mask])

    if upper_bounds is not None:
        mask = u > upper_bounds
        if np.any(mask):
            print("vals exceeding upper bounds")
            print("\tval:", u[mask])
            print("\tupper:", upper_bounds[mask])

    change = change_lower + change_upper

    # TODO don't modify in place for now while testing
    # u += change
    # du += change / alpha
    return u + change, du + change / alpha

class SolverWithWarmStart(CasadiFunctionCallbackMixin, casadi.Callback):

    def __init__(self, name, x, p, g0, g1, lbx, ubx, x0, rootfinder_options, initializer):
        casadi.Callback.__init__(self)
        self.name = name
        self.initializer = initializer
        self.x0 = x0
        self.lbx = np.array(lbx)
        self.ubx = np.array(ubx)

        self.newton = Newton(
            x,
            p,
            g0,
            lbx=lbx,
            ubx=ubx,
            tol=1e-10,
            ls_type=None,
            max_iter=100,
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
        self.x0 = self.initializer(self.x0, p)

        self.x0 = np.array(self.x0).reshape(-1)
        p = p.toarray().reshape(-1)

        lbx_violations = np.where(self.x0 < self.lbx)[0]
        ubx_violations = np.where(self.x0 > self.ubx)[0]
        self.x0[lbx_violations] = self.lbx[lbx_violations]
        self.x0[ubx_violations] = self.ubx[ubx_violations]

        self.x0 = self.newton(self.x0, p)
        self.resid, out_exp = self.resid_func(self.x0, p)
        out_exp = out_exp.toarray().reshape(-1)

        return tuple([*self.x0, *out_exp])

