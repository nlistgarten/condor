import casadi
from condor import assignment_cse, CodePrinter, TableLookup, Solver
from sympy.utilities.lambdify import lambdify

def sympy2casadi(
        sympy_expr,
        sympy_vars, # really, arguments/signature
        #cse_generator=assignment_cse,
        #cse_generator_args=dict(
            extra_assignments={}, return_assignments=True
        #)
):
    ca_vars = casadi.vertcat(*[casadi.MX.sym(s.name) for s in sympy_vars])

    ca_vars_split = casadi.vertsplit(ca_vars)

    mapping = {
        "ImmutableDenseMatrix": casadi.blockcat,
        "MutableDenseMatrix": casadi.blockcat,
        "Abs": casadi.fabs,
        "abs": casadi.fabs,
        "Array": casadi.MX,
    }
    # can set allow_unknown_functions=False and pass user_functions dict of
    # {func_expr: func_str} instead of allow_unknown_function but  it's the
    # presence in "modules" that adds them to namespace on `exec` line of lambdify so
    # it's a little redundant. sticking with this version for now
    # user_functions gets added to known_functions
    printer = CodePrinter(dict(
        fully_qualified_modules=False,
    ))
    print("lambdifying...")
    f = lambdify(
        sympy_vars,
        sympy_expr,
        modules=[
            TableLookup.registry,
            Solver.registry,
            mapping,
            casadi
        ],
        printer=printer,
        dummify=False,
        #cse=cse_generator(**cse_generator_args),
        cse=(
            assignment_cse(extra_assignments, return_assignments)
        ),
    )

    print("casadifying...")
    out = f(*ca_vars_split)

    return out, ca_vars, f

# TODO: update from best results

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

