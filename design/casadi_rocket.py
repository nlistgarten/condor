#
#     This file is part of CasADi.
#
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
#                             K.U. Leuven. All rights reserved.
#     Copyright (C) 2011-2014 Greg Horn
#
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
import casadi
import matplotlib.pyplot as plt

class CasadiNLPWithWarmstart(casadi.Callback):

    def __init__(self, name, *solver_args, cb_opts={}):
        casadi.Callback.__init__(self)

        self.solver = casadi.nlpsol(name + "_solver", *solver_args)

        self.construct(name, cb_opts)

    def get_n_in(self):
        return casadi.nlpsol_n_in()

    def get_n_out(self):
        return casadi.nlpsol_n_out()

    def eval(self, args):
        pass


use_cb_for_jac = True


class StatefulFunction(casadi.Callback):

    def __init__(self, name, func, opts={}):
        casadi.Callback.__init__(self)
        self.func = func
        self.name = name
        self.construct(name, opts)

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
        print(self.name)
        return [out] if self.func.n_out() == 1 else out

    def get_sparsity_in(self, i):
        return self.func.sparsity_in(i)

    def get_sparsity_out(self, i):
        return self.func.sparsity_out(i)

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        print(name, inames, onames, opts)
        if use_cb_for_jac:
            jcb = StatefulFunction(name, self.func.jacobian())
            self.jcb = jcb
            return jcb
        else:
            return self.func.jacobian()


# Control
u = casadi.MX.sym("u")

# State
x = casadi.MX.sym("x", 3)
s = x[0]  # position
v = x[1]  # speed
m = x[2]  # mass

# ODE right hand side
sdot = v
vdot = (u - 0.05 * v * v) / m
mdot = -0.1 * u * u
xdot = casadi.vertcat(sdot, vdot, mdot)

# ODE right hand side function
f = casadi.Function("f", [x, u], [xdot])

# Integrate with Explicit Euler over 0.2 seconds
dt = 0.01  # Time step
xj = x
for j in range(20):
    fj = f(xj, u)
    xj += dt * fj

# Discrete time dynamics function
F = casadi.Function("F", [x, u], [xj])


# Number of control segments
nu = 50

# Control for all segments
U = casadi.MX.sym("U", nu)

# Initial conditions
X0 = casadi.MX([0, 0, 1])

# Integrate over all intervals
X = X0
for k in range(nu):
    X = F(X, U[k])

# Objective function and constraints
J = casadi.mtimes(U.T, U)  # u'*u in Matlab
G_expr = X[0:2]  # x(1:2) in Matlab
G_func = casadi.Function("G_func", [U], [G_expr])
G = StatefulFunction("G", G_func)



# NLP
nlp = {"x": U, "f": J, "g": G(U)}
# Allocate an NLP solver
opts = {"ipopt.tol": 1e-10, "expand": False}

opts = {"expand": False, "ipopt": {"tol": 1e-10, "hessian_approximation":"limited-memory"}}

solver = casadi.nlpsol("solver", "ipopt", nlp, opts)

arg = {}
# Bounds on u and initial condition
arg["lbx"] = -0.5
arg["ubx"] = 0.5
arg["x0"] = 0.0

# Bounds on g
arg["lbg"] = [10, 0]
arg["ubg"] = [10, 0]

# Solve the problem
res = solver(**arg)

# Get the solution
plt.plot(res["x"], label="x")
plt.plot(res["lam_x"], label="lam_x")
plt.legend()
plt.grid()
plt.show()
