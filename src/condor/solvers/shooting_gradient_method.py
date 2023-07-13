import casadi
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import fsolve

from dataclasses import dataclass, field, InitVar
from scikits.odes.sundials.cvode import CVODE, StatusEnum
from collections import namedtuple

Root = namedtuple("Root", ["index", "rootsfound"])

class NextTimeFromSlice:
    def __init__(self, at_time_func):
        self.at_time_func = at_time_func

    def set_p(self, p):
        start, stop, step = self.at_time_func(p)
        self.start = start
        self.step = step
        self.stop = stop
        self.direction = np.sign(step)

    def before_start(self, t):
        if self.direction < 0:
            return t > self.start
        return t < self.start

    def after_stop(self, t):
        # if t is exactly stop time, this has already occured
        if self.direction < 0:
            return t <= self.stop
        return t >= self.stop

    def __call__(self, t):
        if self.before_start(t):
            return self.start
        if self.after_stop(t):
            return self.direction*np.inf
        # TODO handle negative step -- may need to adjust a few of these
        return (1+(t-self.start)//self.step)*self.step + self.start

class TimeGeneratorFromSlices:
    def __init__(self, time_slices, direction=1):
        self.time_slices = time_slices
        self.direction = direction

    def __call__(self, p):
        # TODO handle negative step?
        for time_slice in self.time_slices:
            time_slice.set_p(p)

        t = -self.direction*np.inf
        if self.direction > 0:
            get_time = min
        else:
            get_time = max
        while True:
            next_times = [time_slice(t) for time_slice in self.time_slices]
            t = get_time(next_times)
            yield t
            if np.isinf(t):
                breakpoint()


class System:
    def __init__(
        self, initial_state, dot, jac, time_generator,
        events, updates, num_events, terminating,
    ):

        # simulation must be terminated with event so must provide everything

        # result is a temporary instance attribute so system model can pass information
        # to functions the sundials solvers call -- could be passed via the userdata
        # option but these functions need wrappers to handle returned values anyway
        self.result = None

        # these instance attributes encapsolate the business data of a system

        #   functions, see method wrapper for expected signature

        #     for CVODE interface once wrapped
        self._dot = dot
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

        self.num_events = len(updates)
        self.make_solver()

    def make_solver(self):
        self.solver = CVODE(
            self.dots,
            jacfn=self.jac,
            old_api=False,
            one_step_compute=True,
            rootfn=self.events,
            nr_rootfns=self.num_events,
        )

    def initial_state(self):
        return self._initial_state(self.result.p).toarray().squeeze()

    def dots(self, t, x, xdot,):# userdata=None,):
        xdot[:] = self._dot(self.result.p, t, x).toarray().squeeze()

    def jac(self, t, x, xdot, jac):#
        jac[...] = self._jac(self.result.p, t, x).toarray().squeeze()

    def events(self, t, x, g):
        g[:] = self._events(self.result.p, t, x).toarray().squeeze()

    def update(self, t, x, rootsfound):
        next_x = x # who is responsible for copying? I suppose simulate
        for root_sign, update in zip(rootsfound, self._updates):
            if root_sign != 0:
                # 
                next_x = update(self.result.p, res.values.t, next_x)
        return next_x.toarray().squeeze()

    def time_generator(self):
        for t in self._time_generator(self.result.p):
            yield t.toarray()[0,0]


    def simulate(self):
        """
        expects:
        self.solver is an object with CVODE-like interface to parameterized
        dots, [jac,] and rootfns with init_step, set_options(tstop=...), and step()
        returns StatusEnum with TSTOP_RETURN and ROOT_RETURN members.

        self.result is a namespace with t, x, and e attributes that can be appended with
        time value, state at time, and possibly events e
        initializer 
        update 
        """
        results = self.result
        solver = self.solver
        time_generator = self.time_generator()

        last_t = next(time_generator)
        last_x = self.initial_state()
        results.t.append(last_t)
        results.x.append(last_x)
        rootsfound = self.events(last_t, last_x, np.empty(self.num_events)) == 0.

        # each iteration of this loop simulates until next generated time
        while True:

            next_t = next(time_generator)
            if np.isinf(next_t):
                break
            solver.init_step(last_t, last_x)
            solver.set_options(tstop=next_t)
            last_t = next_t

            # each iteration of this loop simulates until next event or time stop
            while True:
                res = solver.step(next_t)
                if res.flag < 0:
                    breakpoint()

                results.t.append(np.copy(res.values.t))
                results.x.append(np.copy(res.values.y))

                if res.flag == StatusEnum.ROOT_RETURN:
                    idx = len(results.t)
                    rootsfound = solver.rootinfo()
                    results.e.append(Root(idx, rootsfound))

                    next_x = self.update(
                        res.values.t, np.copy(res.values.y), rootsfound

                    )
                    terminate = np.any(rootsfound[self.terminating] != 0)
                    did_update = np.all(next_x == res.values.y)
                    if not did_update or not terminate:
                        results.t.append(np.copy(res.values.t))
                        results.x.append(next_x)

                    if terminate:
                        return

                    solver.init_step(res.values.t, next_x)
                    last_x = next_x

                if res.values.t == next_t or res.flag == StatusEnum.TSTOP_RETURN:
                    break


    def __call__(self, p):
        self.result = Result(p=p, system=self)
        self.simulate()
        result = self.result
        self.result = None
        return result


@dataclass
class ResultMixin:
    p: list[float]


@dataclass
class ResultBase:
    system: System
    t: list[float] = field(default_factory=list)
    x: list[list] = field(default_factory=list)
    e: list[Root] = field(default_factory=list)


@dataclass
class Result(ResultBase, ResultMixin):
    pass



@dataclass
class ResultInterpolant:
    result: Result
    function: InitVar[callable] = lambda p, t, x: x
    # don't pass interpolants to init?
    # should state_Result be saved or just be an initvar? I back-references are OK so 
    # we can keep it...
    interpolants: list[callable] | None = None # field(init=False)
    time_bounds: list[float] | None = None # field(init=False)
    time_comparison: callable = field(init=False) # method
    interval_select: int = field(init=False)
    event_idxs: list | None = None

    def __post_init__(self, function):
        # make interpolants
        result = self.result
        if self.event_idxs is None:
            # currently expect each root.index to be to the right of each event so is
            # the start of each segment, so zero belongs with this set by adding the 
            # element len(result.t), the slice of pairs slice(idx[i], idx[i+1]) 
            # captures the whole segment including the last one. 

            event_idxs = self.event_idxs = np.array([0] + [
                root.index for root in result.e
            ] + [len(result.t)])

        if self.time_bounds is None:

            self.time_bounds = np.empty(self.event_idxs.shape)
            self.time_bounds[:-1] = np.array(result.t)[event_idxs[:-1]]
            self.time_bounds[-1] = result.t[-1]
        if result.t[-1] < result.t[0]:
            self.time_comparison = self.time_bounds.__le__
            self.interval_select = 0
        else:
            self.time_comparison = self.time_bounds.__ge__
            self.interval_select = -1

        if self.interpolants is None:
            self.interpolants = [
                make_interp_spline(
                    result.t[idx0:idx1],
                    [function( result.p, t, x,) 
                     for t, x in zip(result.t[idx0:idx1], result.x[idx0:idx1])],
                    k=min(3, idx1-idx0)
                )
                for idx0, idx1 in zip(
                    event_idxs[:-1],
                    event_idxs[1:],
                )
            ]

    def __call__(self, t):
        interval_idx = np.where(self.time_comparison(t))[0][self.interval_select]
        return self.interpolants[interval_idx](t)



@dataclass
class AdjointResultMixin:
    state_jacobian: ResultInterpolant
    forcing_function: ResultInterpolant
    state_result: Result = field(init=False)

    def __post_init__(self):
        # TODO: validate forcing_function.result is the same?
        self.state_result = self.state_jacobian.result


@dataclass
class AdjointResult(ResultBase, AdjointResultMixin,):
    pass

class AdjointSystem(System):
    def __init__(
            self, state_jac, dte_dxs, d2te_dxdts, dh_dxs,
    ):
        """
        to support caching jacobians, 
        """
        self.state_jac = state_jac

    def time_generator(self):
        last_event = self.result.state_result.e[-1]
        if last_event.index != len(self.result.state_result.t)-1:
            yield self.state_result.t[-1]
        for event in self.result.state_result.e[::-1]:
            self.rootsfound = event.rootsfound
            yield self.state_result.t[event.index]

    def initial_state(self):
        return self.final_lamda

    def __call__(self, final_lamda, forcing_function, state_jacobian):
        self.final_lamda = final_lamda
        self.result = AdjointResult(
            system=self,
            state_jacobian=state_jacobian,
            forcing_function=forcing_function
        )
        self.simulate()
        return super().simulate()

    def dots(self, t, x, xdot,):# userdata=None,):
        # TODO adjoint system takes jacobian and integrand terms, do matrix multiply and
        # vector add here. then can ust call jacobian term for jac method
        # then before simulate, could actually construct 
        # who would own the data for interpolating ? maybe just a new (data) class that
        # also stores the interpolant? then adjointsystem simulate can create it if
        # it'snot provided
        xdot[:] = -self.result.state_jacobian(t).T @ x - self.result.forcing_function(t)

    def jac(self, t, x, xdot, jac, userdata=None,):
        jac[...] = -self.result.state_jacobian(t).T

@dataclass
class TrajectoryAnalysis:
    integrand_terms: callable
    terminal_terms: callable

    def __call__(self, result):
        # evaluate the trajectory analysis of this result
        # should this return a dataclass? Or just the vector of results?
        integral = 0.
        integrand_interpolant = ResultInterpolant(result=result, function=self.integrand_terms)
        for segment_interpolant, t0, t1 in zip(
            integrand_interpolant.interpolants,
            integrand_interpolant.time_bounds[:-1],
            integrand_interpolant.time_bounds[1:],
        ):
            integrand_antideriv = segment_interpolant.antiderivative()
            breakpoint()
            integral += integrand_antideriv(t1) - integrand_antideriv(t0)
        return self.terminal_terms(result.p, result.t[-1], result.x[-1]) + integral


@dataclass
class ShootingGradientMethod:
    # per system
    p_x0_p_params: callable
    p_dots_p_params: callable # could be cached, or just combined with integrand terms
    # per system's events (length number of events)
    d_update_d_params: list[callable]
    d_event_time_d_params: list[callable]
    d2_event_time_d_params_d_t: list[callable]

    # of length number of (trajectory) outputs
    p_integrand_terms_p_params: list[callable]
    p_terminal_terms_p_params: list[callable]
    p_integrand_terms_p_state: list[callable]
    p_terminal_terms_p_state: list[callable]


    def __call__(self, adjoint_result):
        pass


