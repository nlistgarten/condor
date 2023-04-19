import casadi
import casadi_implementations as implementations

name = 'Casadi'
symbol_class = casadi.MX

def symbol_generator(name, n=1, m=1, symmetric=False, diagonal=False):
    print("casadi creating",name, n, m, symmetric, diagonal)
    sym = casadi.MX.sym(name, (n, m))
    #sym = MixedMX.sym(name, (n, m))
    if symmetric:
        assert n == m
        return casadi.tril2symm(casadi.tril(sym))
    if diagonal:
        assert m == 1
        return casadi.diag(sym)
    return sym


# TODO: update from best results (back_track branch)

class CasadiFunctionCallbackMixin:
    """Base class for wrapping a Function with a Callback"""

    def __init__(self, func, opts={}):
        casadi.Callback.__init__(self)
        self.func = func
        self.construct(func.name(), opts)

    def init(self):
        pass

    def finalize(self):
        pass

    def get_n_in(self):
        return self.func.n_in()

    def get_n_out(self):
        return self.func.n_out()

    def eval(self, args):
        out = self.func(*args)
        return [out] if self.func.n_out() == 1 else out

    def get_sparsity_in(self, i):
        return self.func.sparsity_in(i)

    def get_sparsity_out(self, i):
        return self.func.sparsity_out(i)

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        return self.func.jacobian()


class SolverWithWarmStart(CasadiFunctionCallbackMixin, casadi.Callback):

    def __init__(self, name, nlp_args, x0, p0, lbx, ubx, lbg, ubg):
        casadi.Callback.__init__(self)
        self.nlp_args = nlp_args
        self.name = name
        ipopt_opts = {
            "print_time": False,
            "ipopt": {
                #"hessian_approximation": "limited-memory",
                "print_level": 0,  # 0-2: nothing, 3-4: summary, 5: iter table (default)
            },
        }

        self.ipopt = casadi.nlpsol(
            name+'_ipopt',
            "ipopt",
            nlp_args,
            ipopt_opts
        )
        qrsqp_opts = dict(
            qpsol='qrqp',
            qpsol_options=dict(
                print_iter=False,
                error_on_fail=False
            ),
            verbose=False,
            print_iteration=False,
            print_time=False
        )
        self.qrsqp = casadi.nlpsol(
            name+'_qrsqp',
            #"qrsqp",
            'sqpmethod',
            nlp_args,
            qrsqp_opts
        )

        self.x0 = x0
        self.p0 = p0
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg
        self.nx = len(x0)
        self.np = len(p0)

        solver_in = self.solver_in = dict(
            x0=self.x0, p=nlp_args["p"], lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg
        )
        solver_out = self.qrsqp(**solver_in)
        self.func = casadi.Function(
            name+"_placeholder",
            casadi.vertsplit(nlp_args["p"]),
            casadi.vertsplit(solver_out["x"]) + casadi.vertsplit(solver_out["g"][len(self.ubx):]),
        )
        self.construct(name, {})

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        # NOTE: this works because the solution does not depend on on x0; only a
        # function of objective, constraints, and bounds. Objectives and constraints may
        # depend on p (and therefore may take derivative of wrt to p), bounds assumed
        # constant. 
        return self.func.jacobian()

    def eval(self, args):
        # run IPOPT to get close, then run QRSQP on IPOPT's solution to get closer
        # save QRSQP solution
        ipopt_in = dict(
            x0=self.x0,
            p=casadi.vertcat(*args),
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg
        )
        ipopt_out = self.ipopt(**ipopt_in)

        self.solver_in = ipopt_in.copy()

        qrsqp_in = ipopt_in
        qrsqp_in["x0"] = ipopt_out["x"]
        qrsqp_out = self.qrsqp(**qrsqp_in)

        if not self.qrsqp.stats()['success']:
            raise ValueError(f"qrsqp failed to converge for {self.name}")

        self.solver_out = qrsqp_out
        self.x0 = qrsqp_out["x"]

        return tuple([
            *qrsqp_out["x"].toarray().reshape(-1),
            *qrsqp_out["g"].toarray().reshape(-1)[self.nx:]
        ])

