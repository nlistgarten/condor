from condor.backends.casadi.utils import CasadiFunctionCallbackMixin
import casadi
from simupy.systems import DynamicalSystem
import numpy as np
from scipy import interpolate

def simupy_wrapper(f, p):
    return lambda t, state, output=None, **kwargs: f(p, t, state, **kwargs).toarray().reshape(-1)

class ShootingGradientMethod(CasadiFunctionCallbackMixin, casadi.Callback):
    def __init__(self, intermediate):
        casadi.Callback.__init__(self)
        self.name = name = intermediate.model.__name__
        self.func = casadi.Function(
            f"{intermediate.model.__name__}_placeholder",
            [intermediate.p],
            [intermediate.traj_out_expr],
        )
        self.i = intermediate
        self.construct(name, {})

    def has_jacobian(self):
        return False

    def eval(self, args):
        p = casadi.vertcat(*args)
        simupy_kwargs = {k: simupy_wrapper(v, p) for k, v in self.i.simupy_func_kwargs.items()}
        system = DynamicalSystem(**simupy_kwargs, **self.i.simupy_shape_data)
        system.initial_condition = self.i.x0(p).toarray().reshape(-1)
        tf = getattr(self.i.model, 'tf', self.i.model.default_tf)
        self.res = res = system.simulate(tf)
        integrand = interpolate.make_interp_spline(res.t, [
            self.i.traj_out_integrand_func(p, t, x)
            for t, x in zip(res.t, res.x)
        ]).antiderivative()
        integral = integrand(res.t[-1]) - integrand(res.t[0])
        return self.i.traj_out_terminal_term_func(p, res.t[-1], res.x[-1]) + integral,


"""

DblIntLQR.implementation.callback([0.1, 0])

import casadi
from simupy.systems import DynamicalSystem
from condor.backends.casadi.shooting_gradient_method import simupy_wrapper
self = DblIntLQR.implementation.callback
args = ([1., 0.], )

"""
