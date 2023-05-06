from condor.backends.casadi.utils import CasadiFunctionCallbackMixin
import casadi
from simupy.systems import DynamicalSystem
from simupy.block_diagram import DEFAULT_INTEGRATOR_OPTIONS
import numpy as np
from scipy import interpolate

def simupy_wrapper(f, p):
    return lambda t, state, output=None, **kwargs: f(p, t, state, *kwargs.values()).toarray().reshape(-1)


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
        self.int_options = DEFAULT_INTEGRATOR_OPTIONS.copy()
        self.int_options.update(self.i.kwargs)

    def has_jacobian(self):
        return False

    def eval(self, args):
        """
import casadi
from simupy.systems import DynamicalSystem
from condor.backends.casadi.shooting_gradient_method import simupy_wrapper
self = DblIntDtLQR.implementation.callback
args = ([1., 0.], )

        """
        p = casadi.vertcat(*args)
        simupy_kwargs = {k: simupy_wrapper(v, p) for k, v in self.i.simupy_func_kwargs.items()}
        system = DynamicalSystem(**simupy_kwargs, **self.i.simupy_shape_data)
        system.initial_condition = self.i.x0(p).toarray().reshape(-1)
        tf = getattr(self.i.model, 'tf', self.i.model.default_tf)
        self.res = res = system.simulate(tf, integrator_options=self.int_options)
        #event_times, event_channels = np.where(np.diff(np.sign(res.e), axis=0) != 0)
        event_times, event_channels = np.where(
            np.abs(np.diff(np.sign(res.e), axis=0)) == 2
        )
        integral = 0.
        event_times_list = (event_times+1).tolist()
        if len(event_times_list) == 0 or event_times_list[0] != 0:
            event_times_list = [0] + event_times_list
        for t_start_idx, t_end_idx in zip(event_times_list, event_times_list[1:] + [None]):

            segment_slice = slice(t_start_idx, t_end_idx)
            integrand = interpolate.make_interp_spline(res.t[segment_slice], [
                self.i.traj_out_integrand_func(p, t, x)
                for t, x in zip(res.t[segment_slice], res.x[segment_slice])
            ])
            integrand_antideriv = integrand.antiderivative()
            integral += integrand_antideriv(res.t[segment_slice][-1]) - integrand_antideriv(res.t[segment_slice][0])
        return self.i.traj_out_terminal_term_func(p, res.t[-1], res.x[-1]) + integral,


