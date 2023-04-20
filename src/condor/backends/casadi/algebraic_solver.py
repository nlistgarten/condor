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

    def __init__(
        self, name, nlp_args, g_impl, g_expl, x0, p0, lbx, ubx, lbg, ubg, scalers, adders, initializer=None,
        run_ipopt=False, ls_type=None,
    ):
        self.initializer = initializer
        casadi.Callback.__init__(self)
        self.name = name

        # nlpsol's only
        self.run_ipopt = True #run_ipopt
        self.run_newton = False
        self.run_qrsqp = True
        self.run_rootfinder = True

        # newtons only
        self.run_ipopt = False
        self.run_newton = True
        self.run_qrsqp = False
        self.run_rootfinder = True

        self.scalers = scalers
        self.adders = adders
        #self.x0 = x0
        self.x0 = scalers*(x0+adders)

        self.p0 = p0
        self.lbx = scalers*(lbx + adders)
        self.ubx = scalers*(ubx + adders)
        self.nx = len(x0)
        self.np = len(p0)
        self.lbg = scalers*(lbg[:self.nx] + adders)
        self.ubg = scalers*(ubg[:self.nx] + adders)
        self.g_expl = g_expl
        self.g_impl = g_impl

        base_tol = 1e-14
        rootfinder_opts = dict(
            abstol=base_tol*10,
            error_on_fail=False,
            max_iter=100,
            #abstolStep=1e-12,
            print_iteration=False,
            line_search=True,
        )

        self.newton = Newton(
            nlp_args["x"],
            nlp_args["p"],
            g_impl,
            lbx=self.lbx,
            ubx=self.ubx,
            tol=base_tol,
            ls_type=ls_type,
            max_iter=100,
        )

        # TODO: speedups/performance

        # OM is doing something slightly different than us -- they have an rtol which
        # I believe has the effect of loosening the tolerances early when the first
        # iteration of a sub-solver is bad, then tightening as the overall system
        # converges which is a nice effect. It also seems like rootfinder takes a few
        # iterations after backtracking newton on the cycle-level solvers at least,
        # which seems odd.
        # just saw rho for armijo goldstein LS is 0.75 for hbtf cycle-level

        # overall cleanup, debug level controlled printing, etc. more careful memory
        # allocation, etc

        # it seems like norm_2 is much better for line search objective than sum of
        # squares, does the alpha after failed line search do anything? should do
        # performance testing using iteration counts

        # to C? I think it would be a pretty simple modification from the current
        # rootfinder, maybe worth opening as a PR so it can respect bounds




        self.ipopt_opts = ipopt_opts = dict(
            print_time= False,
            ipopt = dict(
                # hessian_approximation="limited-memory",
                print_level=0,  # 0-2: nothing, 3-4: summary, 5: iter table (default)
                tol=1E-14,
                #max_iter=1000,
            ),
            bound_consistency=True,
            #clip_inactive_lam=True,
            #calc_lam_x=False,
            #calc_lam_p=False,
        )
        # additional options from https://groups.google.com/g/casadi-users/c/OdRQKR13R50/m/bIbNoEHVBAAJ
        # to try to get sensitivity from ipopt. so far no...

        qrsqp_opts = dict(
            qpsol='qrqp',
            qpsol_options=dict(
                print_iter=False,
                error_on_fail=False,
            ),
            verbose=False,
            tol_pr=1E-16,
            tol_du=1E-16,
            print_iteration=False,
            print_time=False,
            #hessian_approximation= "limited-memory",
            print_status=False,
        )

        self.nlp_args = dict(
            x=nlp_args["x"],
            p=nlp_args["p"],
            f=casadi.sumsqr(g_impl)/2,
        )
        if g_expl.shape != (0,0):
            self.nlp_args["g"] = g_expl

        self.resid_func = casadi.Function(
            self.name+"_func",
            [nlp_args["x"], nlp_args["p"]],
            [g_impl, g_expl],
        )

        self.rootfinder = casadi.rootfinder(
            self.name,
            "newton",
            self.resid_func,
            rootfinder_opts,
        )
        out_imp, out_exp = self.rootfinder(x0, nlp_args["p"])

        if self.run_qrsqp:
            self.qrsqp = casadi.nlpsol(
                name+'_qrsqp',
                'sqpmethod',
                self.nlp_args,
                qrsqp_opts
            )

        self.ipopt = casadi.nlpsol(
            name+'_ipopt',
            "ipopt",
            self.nlp_args,
            ipopt_opts
        )


        if self.run_rootfinder or self.run_newton:
            self.func = casadi.Function(
                name+"_placeholder",
                casadi.vertsplit(nlp_args["p"]),
                casadi.vertsplit(out_imp/scalers - adders) + casadi.vertsplit(out_exp),
            )

        else:
            self.solver_in = dict(
                x0=self.x0,
                p=nlp_args["p"],
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=-np.inf,
                ubg=np.inf,
            )

            if self.run_qrsqp:
                symbolic_out = self.qrsqp(**self.solver_in)
            else:
                symbolic_out = self.ipopt(**self.solver_in)

            placeholder_output = casadi.vertsplit(
                symbolic_out["x"]/scalers - adders
            )

            if g_expl.shape != (0,0):
                placeholder_output += casadi.vertsplit(symbolic_out["g"])

            self.func = casadi.Function(
                name+"_placeholder",
                casadi.vertsplit(nlp_args["p"]),
                placeholder_output
            )

        self.construct(name, {})

    def eval(self, args):
        # run IPOPT to get close, then run rootfinder to finish

        if self.initializer is not None:
            # maybe numerical ops get lumped ??
            #self.x0 = self.initializer(self.x0, args)
            self.x0 = self.scalers*(self.initializer(self.x0/self.scalers - self.adders, args) + self.adders)

        p = casadi.vertcat(*args)

        lbx_violations = np.where(self.x0 < self.lbx)[0]
        ubx_violations = np.where(self.x0 > self.ubx)[0]
        self.x0[lbx_violations] = self.lbx[lbx_violations]
        self.x0[ubx_violations] = self.ubx[ubx_violations]

        ipopt_in = dict(
            x0=self.x0,
            p=p,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=-np.inf,
            ubg=np.inf,
        )
        if self.run_ipopt:
            self.ipopt_out = self.ipopt(**ipopt_in)
            self.ipopt_stats = self.ipopt.stats()
            self.x0 = self.ipopt_out["x"].toarray().reshape(-1)
            out_exp = self.ipopt_out["g"].toarray().reshape(-1)
            print(f"ipopt {self.name} {self.ipopt_stats['return_status']}, {self.ipopt_stats['iter_count']} iters")

            #if self.ipopt_out["f"] > 1e-12:
            # if not self.ipopt_stats["success"]:
            if ((not self.ipopt_stats["success"]) and (self.ipopt_out["f"] > 1E-20)) or np.any(self.x0 == self.lbx) or np.any(self.x0 == self.ubx):
                print(f"IPOPT failed for {self.name}")
                print("ipopt in")
                print(*list(ipopt_in.items()), sep="\n")
                print("ipopt out")
                print(*list(self.ipopt_out.items()), sep="\n")
                print("ipopt stats")
                print(*list(self.ipopt_stats.items()), sep="\n")
                sys.exit()

        qrsqp_in = dict(
            x0=self.x0,
            p=p,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=-np.inf,
            ubg=np.inf,
        )
        if self.run_qrsqp:
            self.qrsqp_out = self.qrsqp(**qrsqp_in)
            self.x0 = self.qrsqp_out["x"].toarray().reshape(-1)
            out_exp = self.qrsqp_out["g"].toarray().reshape(-1)
            self.qrsqp_stats = self.qrsqp.stats()
            print(f"qrsqp {self.name} {self.qrsqp_stats['return_status']}, {self.qrsqp_stats['iter_count']} iters")
            if ((not self.qrsqp_stats["success"]) and (self.qrsqp_out["f"] > 1E-12)) or np.any(self.x0 == self.lbx) or np.any(self.x0 == self.ubx):
                print(f"QRSQP failed for {self.name}")
                print("ipopt in")
                print(*list(ipopt_in.items()), sep="\n")
                print("qrsqp in")
                print(*list(qrsqp_in.items()), sep="\n")
                print("qrsqp out")
                print(*list(self.qrsqp_out.items()), sep="\n")
                print("qrsqp stats")
                print(*list(self.ipopt_stats.items()), sep="\n")
                sys.exit()

        if self.run_newton:
            print(f"calling Newton from {self.name} ")
            print(type(self.x0), self.x0)
            print(type(p), p)
            self.x0 = self.newton(self.x0, p.toarray().reshape(-1))
            resid, out_exp = self.resid_func(self.x0, p)
            out_exp = out_exp.toarray().reshape(-1)
            newton_resid_norm = np.linalg.norm(resid)
            print(f"newton {self.name}, resid norm = {newton_resid_norm}, {self.newton.iters} iters")
            #return tuple([
            #    #*self.x0.toarray().reshape(-1),
            #    *(self.x0.reshape(-1)/self.scalers - self.adders),
            #    *out_exp.toarray().reshape(-1),
            #])

        if self.run_rootfinder:

            print(f"running rootfinder from {self.name} ")
            old_x0 = self.x0.copy()
            old_out_exp = out_exp.copy()
            self.x0, out_exp = self.rootfinder(self.x0, p)
            self.x0 = self.x0.toarray().reshape(-1)
            out_exp = out_exp.toarray().reshape(-1)
            self.root_stats = self.rootfinder.stats()
            g_out = self.resid_func(self.x0, p)
            rootfinder_resid_norm = np.linalg.norm(g_out[0])
            print(f"rootfinder {self.name} {self.root_stats['return_status']}, {self.root_stats['iter_count']} iters, resid norm = {rootfinder_resid_norm}")

            if (rootfinder_resid_norm > newton_resid_norm) or np.any(self.x0 < self.lbx) or np.any(self.x0 > self.ubx):
                print(f"root finder failed for {self.name}")
                print(*list(ipopt_in.items()), sep="\n")
                print(*list(self.root_stats.items()), sep="\n")
                self.x0 = old_x0
                out_exp = old_out_exp

        return tuple([
            *(self.x0/self.scalers - self.adders),
            *out_exp,
        ])

