from condor.backends.casadi.utils import CasadiFunctionCallbackMixin
import casadi
from simupy.systems import DynamicalSystem
from simupy.block_diagram import DEFAULT_INTEGRATOR_OPTIONS
import numpy as np
from scipy import interpolate


def adjoint_wrapper(f, p, res, segment_slice):
    sliced_t = res.t[segment_slice]
    sliced_t_1 = sliced_t[-1]
    x_interp = interpolate.make_interp_spline(
        sliced_t,
        res.x[segment_slice, :],
    )
    transform_t = lambda t: sliced_t[-1] - t
    return (
        lambda t, adjoint, output=None, **kwargs:
        f(
            p, x_interp(sliced_t_1 - t), sliced_t_1 - t, adjoint, *kwargs.values()
        ).toarray().reshape(-1)
    )

class ShootingGradientMethodJacobian(CasadiFunctionCallbackMixin, casadi.Callback):
    def has_jacobian(self):
        return False

    def __init__(self, shot, name, inames, onames, opts):
        casadi.Callback.__init__(self)
        self.shot = shot
        self.i = shot.i
        self.name = name
        self.inames = inames
        self.onames = onames
        self.opts = opts
        self.construct(name, {})

    def get_n_in(self):
        return 2

    def get_n_out(self):
        return 1

    def get_sparsity_in(self,i):
        shot = self.shot
        p = shot.i.p
        out = shot.i.traj_out_expr
        if i==0: # nominal input
            return casadi.Sparsity.dense(p.shape)
        elif i==1: # nominal output
            return casadi.Sparsity(out.shape)

    def get_sparsity_out(self,i):
        p = self.i.p
        out = self.i.traj_out_expr
        return casadi.Sparsity.dense(out.shape[0], p.shape[0])

    def eval(self, args):
        print("\n"*10, f"eval jacobian for {self.name}", sep="\n")
        print("args",args)
        if hasattr(self.shot, "p"):
            print(f"p={self.shot.p}")
        print(f"o={self.shot.output.toarray()}")

        assert casadi.is_equal(self.shot.p, args[0])
        assert casadi.is_equal(self.shot.res.p, args[0])
        p = args[0]
        o = args[1]
        num_output = o.shape[0]
        num_parameter = p.shape[0]
        jac = casadi.DM(num_output, num_parameter)
        # TODO: eventually, for cases where n > dim_state, compute adjoint for terminal
        # condition of each state, by linearity compute jacobian of desired outputs

        # for each output
            # initialize pJpp = 0 --> store directly in jac? oh this is dphi/dp
            # initialize lamda = dphi/dx
            # "_dot" = d/dt

        # for each segment:
            # build df/dx and df/dp
            # for each output
                # build dL/dx and dL/dp
                # optimize for L = 0?
                # integrate lamda_dot = df/dx.T @ lamda + dL/dx
                # integrate pJpp_dot = df/dp.T @ lamda + dL/dp
                # update lambda and pJpp for each event


        # OR:

        # df/{dx and dp}, built during init
        # for each output:
            # {dL}/{dx, dp} built during init
            # use {dphi}/{dx, dp} to initialize. Need to know if terminated by an event.
            # Does simupy append an update state if needed? is it ever needed? or
            # inherently event occurance for termination -> no update? How to make sure
            # channel found?
        # for each segment:
            # build x(t)
            # for each output
                # integrate lamda_dot = df/dx.T @ lamda + dL/dx
                # integrate pJpp_dot = df/dp.T @ lamda + dL/dp
                # update lambda and pJpp for each event
                # since we'll have (potentially) multiple 

        # OR
        # use simupy event to trigger each segment, provide each update?

        sim_res = self.shot.res
        lamda0s = [
            lamda0_func(p, sim_res.x[-1], sim_res.t[-1])
            for lamda0_func in self.i.lamda0_funcs
        ]

        for i_out, grad0_func in enumerate(self.i.grad0_funcs):
            jac[i_out, :] = grad0_func(p, sim_res.x[-1], sim_res.t[-1])

        rev_event_times_list = sim_res.event_times_list[::-1]
        for t_start_idx, t_end_idx in zip(
            rev_event_times_list,
            [None] + rev_event_times_list[:-1],
        ):
            segment_slice = slice(t_start_idx, t_end_idx)
            # simulation is from t_0 to t_1
            # adjoint prop is from 0 to (t_1 - t_0) representing t_1 to t_0
            # want 0 -> t_1 and (t_1 - t_0) -> t_0
            adjoint_segment_tf = np.diff(sim_res.t[segment_slice][[0, -1]])[0]
            for idx, (lamda0, lamda_dot_func, grad_dot_func,) in enumerate(
                    zip(lamda0s, self.i.lamda_dot_funcs, self.i.grad_dot_funcs)
            ):
                adjoint_sys = DynamicalSystem(
                    state_equation_function = adjoint_wrapper(
                        lamda_dot_func, p, sim_res, segment_slice
                    ),
                    output_equation_function = adjoint_wrapper(
                        grad_dot_func, p, sim_res, segment_slice
                    ),
                    dim_state = self.i.ode_model.state._count,
                    dim_output = num_parameter,
                )
                adjoint_sys.initial_condition = lamda0.toarray().reshape(-1)
                adjoint_res = adjoint_sys.simulate(
                    adjoint_segment_tf, integrator_options=self.shot.int_options
                )
                integrand_interp = interpolate.make_interp_spline(
                    sim_res.t[segment_slice][-1] - adjoint_res.t[::-1],
                    adjoint_res.y[::-1],
                    # adjoint_res.t,
                    # adjoint_res.y[:],
                )
                integrand_antideriv = integrand_interp.antiderivative()
                jac[idx, :] += (
                    integrand_antideriv(sim_res.t[segment_slice][-1]) -
                    integrand_antideriv(sim_res.t[segment_slice][0])
                ).reshape((1,-1))
            self.res = adjoint_res


        print('computed jac:',jac.toarray())
        return jac,


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

    def get_jacobian(self, name, inames, onames, opts):
        print("\n"*10, f"getting jacobian for {self} . {name}", sep="\n")
        if hasattr(self, "p"):
            print(f"p={self.p}")
        self.jac_callback = ShootingGradientMethodJacobian(self, name, inames, onames, opts)
        return self.jac_callback

    def eval(self, args):
        """
import casadi
from simupy.systems import DynamicalSystem
from condor.backends.casadi.shooting_gradient_method import simupy_wrapper
self = DblIntDtLQR.implementation.callback
args = ([1., 0.], )

        """
        #if isinstance(args, (tuple, list)):
        #    if len(args) == 1 and isinstance(args[0], casadi.DM):
        #        args = args[0]
        #    else:
        #        raise ValueError

        p = casadi.vertcat(*args)
        self.p = p
        simupy_kwargs = {k: simupy_wrapper(v, p) for k, v in self.i.sim_func_kwargs.items()}
        self.simupy_kwargs = simupy_kwargs
        system = DynamicalSystem(**simupy_kwargs, **self.i.sim_shape_data)
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

        self.res.event_times_list = event_times_list
        self.res.event_channels = event_channels
        self.res.p = p

        for t_start_idx, t_end_idx in zip(event_times_list, event_times_list[1:] + [None]):

            segment_slice = slice(t_start_idx, t_end_idx)
            integrand = interpolate.make_interp_spline(res.t[segment_slice], [
                self.i.traj_out_integrand_func(p, t, x)
                for t, x in zip(res.t[segment_slice], res.x[segment_slice])
            ])
            integrand_antideriv = integrand.antiderivative()
            integral += integrand_antideriv(res.t[segment_slice][-1]) - integrand_antideriv(res.t[segment_slice][0])
        self.output = self.i.traj_out_terminal_term_func(p, res.t[-1], res.x[-1]) + integral
        return self.output,


