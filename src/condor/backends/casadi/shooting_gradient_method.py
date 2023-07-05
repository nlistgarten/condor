from condor.backends.casadi.utils import CasadiFunctionCallbackMixin
import casadi
import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve

from dataclasses import dataclass, fields
from scikits.odes.sundials.cvode import CVODE, StatusEnum
from itertools import count

DEBUG_LEVEL = 1


Root = namedtuple("Root", ["index", "rootsfound"])
#Results = namedtuple("Results", ["t", "x", "e", "p"],)


class NextTimeFromSlice:
    def __init__(self, at_time_func):
        self.at_time_func = at_time_func

    def set_p(self, p):
        start, stop, step = self.at_time_func(self.p)
        self.start = start
        self.step = step
        self.stop = stop

    def __call__(self, t):
        # TODO handle negative step?
        if t < self.start:
            return start
        if t >= self.stop:
            # if t is exactly stop time, this has already occured
            return np.inf
        return (1+(t-self.start)//self.step)*self.step + self.start

"""
want to be able to simulate adjoint in segments to prevent re-computing jacobian of
costate, but currently not even caching it! But let's assume we'll need to do that
eventually, so System(Base?) simulate should control events? 

want system to simulate =

really problem is I don't love that we need time_generator_data for System and it has a
different data type and method for AdjointSystem. 
"""

class TimeGeneratorFromSlices:
    def __init__(self, t0, at_time_slices):
        self.t0 = t0
        self.time_slices = time_slices

    def __call__(self, p, t0 = 0.):
        # TODO handle negative step?
        for time_slice in self.time_slices:
            time_slice.set_p(p)
        t  = t0
        while True:
            next_times = [time_slice(t) for time_slice in time_slices]
            t = min(next_times)
            yield t
            if np.isinf(t):
                breakpoint()

class System:
    def __init__(
            self, dots, jac, events, updates, time_generator, initial_state, terminating=[]
    ):
        # p is a temporary instance attribute so system model can pass information
        # correctly -- the Result of each simulation is the true "owner" of p. May try
        # to figure out how to pass the Result around instead
        self.p = None 

        # these instance attributes encapsolate the business data of a system

        #   functions, see method wrapper for expected signature

        #     for CVODE interface once wrapped
        self._dots = dots
        self._jac = jac
        self._events = events
        #     list of functions for 
        self._updates = updates

        #     define initial conditions
        self._initial_state = initial_state
        # who owns t0? time generator? for adjoint system, very easy to own all of them.
        # I guess can just handle single point as a special case instead of assuming all
        # take the form of an interval? Does this make it easier to allow events that
        # occur at t0? Then 

        # data for time_generator method -- should this just be a generator class
        # itself? yes, basically just a sub-name space to the System which should own
        # the (parameterized) callables. Use wrapper method to define interface, then 
        # AdjointSystem can re-implement wrapper method to change interface
        self._time_generator = time_generator

        # list of root indices that are terminating events...
        # any(rootsfound[terminating]) --> terminates simulation
        self.terminating = terminating

        self.make_solver()

    def make_solver(self):
        self.solver = CVODE(
            self.dots,
            jacfn=self.jac,
            old_api=False,
            one_step_compute=True,
            rootfn=self.events,
            nr_rootfns=len(self.updates),
        )

    def initial_state(self):
        return self._initial_state(self.p)

    def dots(self, t, x, xdot,):# userdata=None,):
        xdot[:] = self._dots(self.p, t, x)

    def jac(self, t, x, xdot, jac):#
        jac[...] = self._jac(self.p, t, x)

    def events(self, t, x, g):
        g[:] = self._events(self.p, t, x)

    def update(self, t, x, rootsfound):
        next_x = x # who is responsible for copying? I suppose simulate
        for root_sign, update in zip(rootsfound, self._updates):
            if root_sign != 0:
                # 
                next_x = update(self.p, res.values.t, next_x)
        return next_x

    def time_generator(self):
        for t in self.time_data:
            yield t


    def simulate(self, p, results=None):
        """
        expects:
        self.solver is an object with CVODE-like interface to parameterized
        dots, [jac,] and rootfns with init_step, set_options(tstop=...), and step()
        returns StatusEnum with TSTOP_RETURN and ROOT_RETURN members.

        initializer 
        update 
        """
        if results is None:
            results = Result(p=p, system=self)
        else:
            # validate Result parameters and system?
            pass

        solver = self.solver

        last_t = t0
        next_x = x0

        results.t.append(last_t)
        results.x.append(next_x)

        for next_t in self.time_generator(p, t0):
            if np.isinf(next_t):
                break
            solver.init_step(last_t, next_x)
            solver.set_options(tstop=next_t)
            last_t = next_t
            while True:
                idx = len(results.t)
                res = solver.step(t1)
                print(idx, res.flag, res.values.t)
                if res.flag < 0:
                    breakpoint()

                if res.flag != StatusEnum.TSTOP_RETURN:
                    results.t.append(np.copy(res.values.t))
                    results.x.append(np.copy(res.values.y))

                if res.flag == case StatusEnum.ROOT_RETURN:
                    rootsfound = solver.rootinfo()
                    results.e.append(Root(idx, rootsfound))

                    if np.any(rootsfound[self.terminating] != 0):
                        self.p = None
                        return results

                    next_x = self.update(
                        res.values.t, np.copy(results.x[-1]), rootsfound
                    )
                    results.t.append(np.copy(res.values.t))
                    results.x.append(next_x)
                    solver.init_step(res.values.t, next_x)

                if res.values.t == next_t or res.flag != StatusEnum.TSTOP_RETURN:
                    break

        self.p = None
        return results


@dataclass
class ResultMixin:
    p: list[float]


@dataclass
class ResultBase:
    system: System
    t: list[float] = field(default_factory=list)
    x: list[list] = field(default_factory=list)
    e: list[Root] = field(default_factory=list)

    def __getitem__(self, key):
        return self.__class__(
            *tuple(getattr(self, field.name) for field in fields(self)[:-3]),
            t=self.t[key], x=self.x[key], e=self.e[key]
        )


@dataclass
class Result(ResultMixin, ResultBase):
    pass


@dataclass
class AdjointResultMixin:
    state_result: Result

    # does interpolant belong here or on its own dataclass?
    interpolant: callable = None

    def __post_init__(self):
        if self.interpolant is None:
            pass
            # make interpolants


@dataclass
class AdjointResult(AdjointResultMixin, ResultBase):
    pass


class AdjointSystem(System):
    def __init__(
            self, dots, jac, events, updates, time_generator, terminating=[], p=None,
    ):
        """
        to support caching jacobians, 
        """
        super().__init__(dots, jac, events, updates, self._time_generator,
                         terminating=[0])

    def self.time_generator(self):
        return [t1]

    def initial_state(self):
        return self._initial_state(self.p,)

    def simulate(self, some_arg_to_set_lamda0, state_results=None, adjoint_results=None):
        self.some_arg_to_set_lamda0 = some_arg_to_set_lamda0
        # can also 

        self.x_interp = interpolate.make_interp_spline(
            state_results.t,
            state_results.x,
            k=min(3, len(state_results.t)-1)
        )
        return super().simulate(lamda0, state_results.p, state_results.t[-1], lamda0, sys_res.t[0])

    def dots(self, t, x, xdot,):# userdata=None,):
        # TODO adjoint system takes jacobian and integrand terms, do matrix multiply and
        # vector add here. then can ust call jacobian term for jac method
        # then before simulate, could actually construct 
        # who would own the data for interpolating ? maybe just a new (data) class that
        # also stores the interpolant? then adjointsystem simulate can create it if
        # it'snot provided
        xdot[:] = self._dots(self.p, self.x_interp(t), t, x)

    def jac(self, t, x, xdot, jac, userdata=None,):
        jac[...] = self._jac(self.p, self.x_interp(t), t, x)


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



                jac[idx, :] += lamda_te_p[None, :] @ (
                    self.i.dh_dps[event_channel](p, tem, xtem, event_channel)
                    + 1*(
                        delta_fs 
                    ) @ self.i.dte_dps[event_channel](p, tem, xtem)
                    - delta_xs[:, None] @ self.i.d2te_dpdts[event_channel](p, tem, xtem).T
                ) + 1*(
                    (
                        +1*adjoint_sys.state_equation_function(tep, lamda_te_p)[None, :]
                        -1*adjoint_sys.state_equation_function(tem, lamda_te_m)[None, :]
                    ) @ (delta_xs) @ self.i.dte_dps[event_channel](p, tem, xtem)
                ) - 1*(
                    (lamda_te_p - lamda_te_m)[None, :]  @ self.i.dh_dps[event_channel](p, tem, xtem, event_channel)
                    @ (self.i.dte_dps[event_channel](p, tem, xtem).T @ self.i.dte_dps[event_channel](p, tem, xtem))
                )

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
        p = casadi.vertcat(*args)
        if DEBUG_LEVEL > 1:
            print("shot args:", p.toarray().squeeze())
        self.p = p
        simupy_kwargs = {k: simupy_wrapper(v, p) for k, v in self.i.sim_func_kwargs.items()}
        self.simupy_kwargs = simupy_kwargs
        system = DynamicalSystem(**simupy_kwargs, **self.i.sim_shape_data)
        system.initial_condition = self.i.x0(p).toarray().reshape(-1)

        # TODO: tf could be a parameter, then use model_instance to access tf which
        # should be packed with value. But then need conditional for applying effect of
        # event time on jacobian? So use terminating event -- could automatically
        # generate and new "exact time" would handle it
        tf = getattr(self.i.model, 'tf', self.i.model.default_tf)

        res = SimulationResult(
            **self.i.sim_shape_data,
            tspan=np.array((0., tf)),
        )
        num_events = self.i.sim_shape_data['num_events']
        ts = [0.,] + [
            fsolve(lambda t: simupy_kwargs['event_equation_function'](t, [])[idx], x0=0.)[0]
            if self.i.event_time_only[idx]
            else np.inf
            for idx in range(num_events)
        ]

        terminating = np.array([
            getattr(e, 'terminate', False)
            for e in self.i.ode_model.Event.subclasses
        ], dtype=bool)
        exact_terminating_channels = terminating & np.array(self.i.event_time_only, dtype=bool)
        any_exact_terminating = num_events and np.any(exact_terminating_channels)
        if not any_exact_terminating:
            ts += [tf]

        sorted_t_idx = np.argsort(ts)

        int_options = self.int_options.copy()
        if DEBUG_LEVEL:
            print(ts, sorted_t_idx)
        #for idx, t0, t1 in zip(range(len(ts)), ts, ts[1:]):
        for t0_idx, t1_idx in zip(sorted_t_idx, sorted_t_idx[1:]):
            t0 = ts[t0_idx]
            t1 = ts[t1_idx]
            if np.isinf(t1):
                break
            #half_span = (t1 - t0)/4
            #if self.int_options['max_step'] == 0 or self.int_options['max_step'] > half_span:
            #    int_options['max_step'] = half_span
            system.simulate((t0, t1), integrator_options=int_options, results=res)
            if DEBUG_LEVEL:
                print(f"simulated from idx {t0_idx} at time t={ts[t0_idx]} to idx {t1_idx} at time t={ts[t1_idx]}")
            if t1_idx != sorted_t_idx[-1]:
                if DEBUG_LEVEL:
                    print(f"updating with event channel [{t1_idx-1}]")
                update = system.update_equation_function(
                    res.t[-1], res.x[-1, :], event_channels=[t1_idx-1]
                )
                if np.any(np.isnan(update)):
                    # TODO: is this a bug in casadi? in orbital3 with a single
                    # measurement, get nans for delta_v_disp sometimes as optimizer is
                    # jumping back and forth over discontinuity, usually when burn is
                    # right before measurement, within 1E-6 eg 850.9999999020214 vs 851
                    update[np.isnan(update)] = res.x[-1, np.isnan(update)]
                system.initial_condition = update


        self.res = res
        sign_e = np.sign(res.e)
        #sign_e[terminating[None, :] & np.isclose(res.e, 0., atol=1E-15, rtol=0.)] = 0.
        sign_e[
            exact_terminating_channels[None, :]
            & np.isclose(res.e, 0., atol=1E-15, rtol=0.)
        ] = 0.
        event_select = (
            (np.abs(np.diff(sign_e, axis=0, prepend=-sign_e[[0]])) == 2.)
            | (np.abs(np.diff(sign_e, axis=0, append=sign_e[[-1]])) == 2.)
            | (sign_e == 0.)
        )
        event_select[-1, :] |= terminating & np.isclose(res.e[-1], 0.)

        event_times, event_channels = np.where(event_select)
        event_channels = event_channels[num_events::2]

        if len(event_times) == 0:
            event_times = [0]

        if event_times[-1] != res.t.size-1:
            # this should only occur because simulation ended due to tf, not a
            # terminating event.
            event_times = np.concatenate((event_times, [res.t.size-1]))

        try:
            span_idxs = event_times[max(num_events-1, 0):].reshape(-1,2) + np.array([[0,1]])
        except Exception as e:
            print(e)
            print(res.t[event_times[:-1]], res.t[-1])
            print(res.t[event_times[max(num_events-1, 0):-1]], res.t[-1])
            breakpoint()

        if np.any(np.diff(res.t[span_idxs[:-1]], axis=1)==0.):
            breakpoint()

        if DEBUG_LEVEL > 1:
            print(res.t[span_idxs[:-1]])
            print(res.t[span_idxs[-1,0]:])
        integral = 0.

        self.res.span_idxs = span_idxs
        self.res.event_channels = event_channels
        self.res.event_select = event_select
        self.res.p = p
        res.ts = ts

        #for t_start_idx, t_end_idx in zip(event_times_list, event_times_list[1:] + [None]):

        for span_idx in span_idxs:

            segment_slice = slice(*span_idx)
            try:
                integrand = interpolate.make_interp_spline(res.t[segment_slice], [
                    self.i.traj_out_integrand_func(p, t, x)
                    for t, x in zip(res.t[segment_slice], res.x[segment_slice])
                ], k=min(3, res.t[segment_slice].size-1))
            except Exception as e:
                print(e)
                breakpoint()
            integrand_antideriv = integrand.antiderivative()
            integral += integrand_antideriv(res.t[segment_slice][-1]) - integrand_antideriv(res.t[segment_slice][0])
        self.output = self.i.traj_out_terminal_term_func(p, res.t[-1], res.x[-1]) + integral


        if not self.from_implementation:
            pass

        return self.output,

