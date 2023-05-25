from condor.backends.casadi.utils import CasadiFunctionCallbackMixin
import casadi
from simupy.systems import DynamicalSystem
from simupy.block_diagram import DEFAULT_INTEGRATOR_OPTIONS
import numpy as np
from scipy import interpolate

DEBUG_LEVEL = 1


def adjoint_wrapper(f, p, res, segment_slice):
    x_interp = interpolate.make_interp_spline(
        res.t[segment_slice],
        res.x[segment_slice, :],
    )
    return (
        lambda t, adjoint, output=None, **kwargs:
        f(
            p, x_interp(t), t, adjoint, *kwargs.values()
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
        if DEBUG_LEVEL:
            print("\n"*10, f"eval jacobian for {self.name}", sep="\n")
            print("args",args)
            if hasattr(self.shot, "p"):
                print(f"p={self.shot.p}")
        if DEBUG_LEVEL > 2:
            print(f"o={self.shot.output.toarray()}")

        if not casadi.is_equal(self.shot.p, args[0]):
            self.shot(args[0])
        assert casadi.is_equal(self.shot.res.p, args[0])
        p = args[0]
        o = args[1]
        num_output = o.shape[0]
        num_parameter = p.shape[0]
        jac = casadi.DM(num_output, num_parameter)
        # TODO: eventually, for cases where n > dim_state, compute adjoint for terminal
        # condition of each state, by linearity compute jacobian of desired outputs

        sim_res = self.shot.res
        lamda0s = [
            lamda0_func(p, sim_res.x[-1], sim_res.t[-1]).toarray().reshape(-1)
            for lamda0_func in self.i.lamda0_funcs
        ]
        for i_out, grad0_func in enumerate(self.i.grad0_funcs):
            jac[i_out, :] = grad0_func(p, sim_res.x[-1], sim_res.t[-1])
        if DEBUG_LEVEL > 2:
            print("initial jacobian:", jac.toarray())

        # TODO: make sure this is robust enough:
        terminal_event_channels = np.isclose(sim_res.e[-1], 0.)
        for event_channel in np.where(terminal_event_channels)[0]:
            if not getattr(
                self.i.ode_model.Event.subclasses[event_channel], 'terminate', False
            ):
                continue
            for idx, lamda0, in enumerate(lamda0s):
                tf = sim_res.t[-1]
                xf = sim_res.x[-1]
                fxf = self.i.sim_func_kwargs["state_equation_function"](p, tf, xf)

                lamda0s[idx] += (
                    (
                        self.i.dte_dxs[event_channel](p, tf, xf).T @ fxf.T
                        #- self.i.d2te_dxdts[event_channel](p, tf, xf).T @ xf.T
                    ) @ lamda0
                ).toarray().reshape(-1)

                # doesn't really affect state switch, which is only one that should
                # matter
                use_lamda = lamda0[None, :]
                use_lamda = lamda0s[idx][None, :]


                #breakpoint()
                jac[idx, :] += -use_lamda @ (
                    fxf @ self.i.dte_dps[event_channel](p, tf, xf)
                    #- xf[:, None] @ self.i.d2te_dpdts[event_channel](p, tf, xf).T
                )

        if DEBUG_LEVEL > 2:
            print("terminal event:", jac.toarray())


        rev_event_times_list = sim_res.event_times_list[::-1]
        for t0_idx, t1_idx, event_channel, in zip(
            rev_event_times_list,
            [None] + rev_event_times_list[:-1],
            sim_res.event_channels[::-1].tolist() + [None],
        ):
            segment_slice = slice(t0_idx, t1_idx)
            adjoint_int_opts = self.shot.int_options.copy()
            tspan = sim_res.t[segment_slice][[-1, 0]]
            adjoint_t_duration = -np.diff(tspan)[0]
            if adjoint_t_duration < adjoint_int_opts['max_step']*2:
                adjoint_int_opts['max_step'] = adjoint_t_duration/4
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
                    num_events=0,
                )
                adjoint_sys.initial_condition = lamda0
                try:
                    adjoint_res = adjoint_sys.simulate(
                        tspan,
                        integrator_options=adjoint_int_opts
                    )
                except Exception as e:
                    print(
                        "getting error during adjoint simulation",
                        f"event_channel: {event_channel}  output idx: {idx}"
                    )
                    raise e
                try:
                    integrand_interp = interpolate.make_interp_spline(
                        adjoint_res.t[::-1],
                        adjoint_res.y[::-1],
                        k=min(3, adjoint_res.t.size-2),
                    )
                except Exception as e:
                    print(e)
                    breakpoint()
                integrand_antideriv = integrand_interp.antiderivative()
                # TOGGLE THIS ONE
                #breakpoint()
                jac[idx, :] += (
                    integrand_antideriv(tspan[0]) -
                    integrand_antideriv(tspan[1])
                ).reshape((1,-1))

                """
                ct_sim = DblIntLQR([0., 0.,])
                ct_sim.implementation.callback.jac_callback([0., 0.,], [0.])

                dt_sim = DblIntLQR([0., 0.,])
                ct_sim.implementation.callback.jac_callback([0., 0.,], [0.])
                """

                lamdaf = adjoint_res.x[-1]

                if event_channel is None:
                    if t0_idx != 0:
                        raise ValueError

                    jac[idx, :] += lamdaf[None, :] @ self.i.dx0_dp(p)


                    continue

                tep = sim_res.t[t0_idx]
                xtep = sim_res.x[t0_idx]
                #xtep = sim_res.t[t0_idx]
                tem_idx = t0_idx-1

                tem = sim_res.t[tem_idx]
                xtem = sim_res.x[tem_idx]

                delta_fs = (
                    self.i.sim_func_kwargs["state_equation_function"](p, tep, xtep)
                    - self.i.sim_func_kwargs["state_equation_function"](p, tem, xtem)
                )

                delta_xs = xtep - xtem

                #print("xtep:", xtep, "delta_fs", delta_fs, "delta_xs", delta_xs)


                lamda0s[idx] = ((
                    self.i.dh_dxs[event_channel](p, tem, xtem, event_channel).T
                    - self.i.dte_dxs[event_channel](p, tem, xtem).T @ (
                        delta_fs.T #+ lam_dot[None, :]
                    )
                    + self.i.d2te_dxdts[event_channel](p, tem, xtem).T @ delta_xs.T
                ) @ lamdaf).toarray().reshape(-1)

                lamda_te_p = lamdaf
                lamda_te_m = lamda0s[idx]
                if self.i.use_lam_te_p:
                    # correct for sampled LQR
                    use_lamda = lamda_te_p[None, :]
                else:
                    # dramatically improves state-switch but incorrect for sampled LQR
                    # state-switch not actually good
                    # but really good for time switched!!!
                    use_lamda = lamda_te_m[None, :]

                if self.i.include_lam_dot:
                    lam_dot = adjoint_sys.state_equation_function(tep, lamdaf)
                else:
                    lam_dot = 0.


                #breakpoint()
                jac[idx, :] += use_lamda @ (
                    self.i.dh_dps[event_channel](p, tem, xtem, event_channel)
                    + (
                        delta_fs 
                    ) @ self.i.dte_dps[event_channel](p, tem, xtem)
                    - delta_xs[:, None] @ self.i.d2te_dpdts[event_channel](p, tem, xtem).T
                ) - (
                    (1*adjoint_sys.state_equation_function(tem, lamda_te_m)[None, :] -
                    1*adjoint_sys.state_equation_function(tep, lamda_te_p)[None, :]) @
                    (delta_xs) @ self.i.dte_dps[event_channel](p, tem, xtem)
                ) - (
                    (lamda_te_p - lamda_te_m)[None, :]  @ self.i.dh_dps[event_channel](p, tem, xtem, event_channel)
                    @ (self.i.dte_dps[event_channel](p, tem, xtem).T @ self.i.dte_dps[event_channel](p, tem, xtem))
                )

            if DEBUG_LEVEL > 2:
                print("event", event_channel, jac.toarray())

            self.res = adjoint_res

        if DEBUG_LEVEL > 2:
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
        self.int_options.update(self.i.integrator_options)

    def get_jacobian(self, name, inames, onames, opts):
        if DEBUG_LEVEL:
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

        # TODO: tf could be a parameter, then use model_instance to access tf which
        # should be packed with value. But then need conditional for applying effect of
        # event time on jacobian? So use terminating event
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


