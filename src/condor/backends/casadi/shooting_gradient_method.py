from condor.backends.casadi.utils import CasadiFunctionCallbackMixin
import casadi
import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve
from condor.solvers.shooting_gradient_method import ResultInterpolant


DEBUG_LEVEL = 1

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
        p = args[0]
        o = args[1]
        jac = self.i.shooting_gradient_method(self.shot.res)
        return jac,


        if DEBUG_LEVEL > 1:
            print("\n"*10, f"eval jacobian for {self.name}", sep="\n")
            print("args",args)
            if hasattr(self.shot, "p"):
                print(f"p={self.shot.p.toarray().squeeze()}")
        if DEBUG_LEVEL > 2:
            print(f"o={self.shot.output.toarray().squeeze()}")

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
            print("initial lamda:", lamda0s)

        terminal_event_channels = sim_res.event_select[-1]
        for event_channel in np.where(terminal_event_channels)[0]:
            if not getattr(
                self.i.ode_model.Event.subclasses[event_channel], 'terminate', False
            ):
                continue
            for idx, lamda0, in enumerate(lamda0s):
                tf = sim_res.t[-1]
                xf = sim_res.x[-1]
                fxf = self.i.sim_func_kwargs["state_equation_function"](p, tf, xf)

                jac[idx, :] += -lamda0[None, :] @ (
                    fxf @ self.i.dte_dps[event_channel](p, tf, xf)
                )

                lamda0s[idx] -= (
                    (
                        self.i.dte_dxs[event_channel](p, tf, xf).T @ fxf.T
                    ) @ lamda0
                ).toarray().reshape(-1)


        if DEBUG_LEVEL > 2:
            print("terminal event:", jac.toarray())
            print("tf- lamda:", lamda0s)
            print("fxf", fxf)
            print("dte dx", self.i.dte_dxs[event_channel](p, tf, xf).T)

        for (t0_idx, t1_idx), event_channel, in zip(
            sim_res.span_idxs[::-1],
            sim_res.event_channels[-1-np.any(terminal_event_channels)::-1].tolist() + [None],
        ):
            segment_slice = slice(t0_idx, t1_idx)
            adjoint_int_opts = self.shot.int_options.copy()
            tspan = sim_res.t[segment_slice][[-1, 0]]
            adjoint_t_duration = -np.diff(tspan)[0]
            #if (adjoint_t_duration < adjoint_int_opts['max_step']*2:
            # this helps orbital_3 which has an event druation -> 0
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
                jac[idx, :] += (
                    integrand_antideriv(tspan[0]) -
                    integrand_antideriv(tspan[1])
                ).reshape((1,-1))

                lamdaf = adjoint_res.x[-1]

                # handle final segment for initial conditions
                if event_channel is None:
                    if t0_idx != 0:
                        breakpoint()
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
                    - self.i.dte_dxs[event_channel](p, tem, xtem).T @ delta_fs.T
                    + self.i.d2te_dxdts[event_channel](p, tem, xtem).T @ delta_xs.T
                ) @ lamdaf
                - 1*(
                    (
                        1*adjoint_sys.state_equation_function(tep, lamdaf)[None, :]
                    ) @ (delta_xs) @ self.i.dte_dxs[event_channel](p, tem, xtem)
                ).T

                                ).toarray().reshape(-1)

                lamda_te_p = lamdaf
                lamda_te_m = lamda0s[idx]



                #jac[idx, :] += 
                if event_channel==2 and False:
                    print("event", event_channel)#, jac.toarray())
                    print("measurement time", self.shot.p[-1])
                    print("delta fs")
                    print(delta_fs)
                    print("delta xs")
                    print(delta_xs)
                    print("dte dps")
                    print(self.i.dte_dps[event_channel](p, tem, xtem))
                    print("dh dps")
                    print(self.i.dh_dps[event_channel](p, tem, xtem, event_channel))
                    print("lamda te m")

                    print("lamda dot -")
                    print(adjoint_sys.state_equation_function(tem, lamda_te_m)[None, :])
                    print("lamda dot +")
                    print(adjoint_sys.state_equation_function(tep, lamda_te_p)[None, :])
                    print( lamda_te_m)


            self.res = adjoint_res

        if DEBUG_LEVEL > 1:
            print('computed jac:',jac.toarray())


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
            dict(allow_free=True,),
        )
        self.i = intermediate
        self.construct(name, {})
        self.from_implementation = False

    def get_jacobian(self, name, inames, onames, opts):
        if DEBUG_LEVEL:
            print("\n"*10, f"getting jacobian for {self} . {name}", sep="\n")
            if hasattr(self, "p"):
                print(f"p={self.p}")
        self.jac_callback = ShootingGradientMethodJacobian(self, name, inames, onames, opts)
        return self.jac_callback

    def eval(self, args):
        p = self.p = casadi.vertcat(*args)
        self.res = self.i.StateSystem(p)
        self.output = self.i.trajectory_analysis(self.res)
        return self.output,

