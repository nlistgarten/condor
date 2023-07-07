import casadi
import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve

from dataclasses import dataclass, fields
from scikits.odes.sundials.cvode import CVODE, StatusEnum
from itertools import count

Root = namedtuple("Root", ["index", "rootsfound"])

class NextTimeFromSlice:
    def __init__(self, at_time_func):
        self.at_time_func = at_time_func

    def set_p(self, p):
        start, stop, step = self.at_time_func(self.p)
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
        # TODO handle negative step?
        if self.before_start(t):
            return self.start
        if self.after_stop(t):
            return self.direction*np.inf
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
            next_times = [time_slice(t) for time_slice in time_slices]
            t = get_time(next_times)
            yield t
            if np.isinf(t):
                breakpoint()


class System:
    def __init__(
            self, dots, jac, events, updates, time_generator, initial_state, terminating=[]
    ):
        # result is a temporary instance attribute so system model can pass information
        # correctly 
        self.result = None

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
        return self._initial_state(self.result.p)

    def dots(self, t, x, xdot,):# userdata=None,):
        xdot[:] = self._dots(self.result.p, t, x)

    def jac(self, t, x, xdot, jac):#
        jac[...] = self._jac(self.result.p, t, x)

    def events(self, t, x, g):
        g[:] = self._events(self.result.p, t, x)

    def update(self, t, x, rootsfound):
        next_x = x # who is responsible for copying? I suppose simulate
        for root_sign, update in zip(rootsfound, self._updates):
            if root_sign != 0:
                # 
                next_x = update(self.result.p, res.values.t, next_x)
        return next_x

    def time_generator(self):
        for t in self._time_generator(self.result.p):
            yield t


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
                res = solver.step(t1)
                print(idx, res.flag, res.values.t)
                if res.flag < 0:
                    breakpoint()

                if res.flag != StatusEnum.TSTOP_RETURN:
                    results.t.append(np.copy(res.values.t))
                    results.x.append(np.copy(res.values.y))

                if res.flag == case StatusEnum.ROOT_RETURN:
                    idx = len(results.t)
                    rootsfound = solver.rootinfo()
                    results.e.append(Root(idx, rootsfound))

                    if np.any(rootsfound[self.terminating] != 0):
                        return

                    next_x = self.update(
                        res.values.t, np.copy(results.x[-1]), rootsfound
                    )
                    results.t.append(np.copy(res.values.t))
                    results.x.append(next_x)
                    solver.init_step(res.values.t, next_x)
                    last_x = next_x

                if res.values.t == next_t or res.flag != StatusEnum.TSTOP_RETURN:
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
class Result(ResultMixin, ResultBase):
    pass


@dataclass
class AdjointResultMixin:
    state_result: Result


@dataclass
class AdjointResult(AdjointResultMixin, ResultBase):
    pass


@dataclass
class ResultInterpolantMixin:
    function: callable = lambda p, t, x: x
    interpolants: list[callable] = None # field(init=False)
    time_bounds: list[float] = None # field(init=False)
    event_idxs: list

    def __post_init__(self):
        # make interpolants
        event_idxs = [root.index for root in self.state_result.e]
        self.time_bounds = np.array(self.state_result.t)[
            [0] + event_idxs + [-1]
        ]
        self.interpolants = [
            make_interp_spline(
                self.state_result.t[idx0:idx1],
                self.function(
                    self.state_result.p,
                    self.state_result.t[idx0:idx1],
                    self.state_result.x[idx0:idx1],
                ),
                k=min(3, idx1-idx0)
        pass

    def __call__(self, t):
    


@dataclass
class ResultInterpolant(AdjointResultMixin, ResultInterpolantMixin):
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
        self.

    def initial_state(self):
        return self._initial_state(self.p, self.some_arg_to_set_lamda0)

    def __call__(self, some_arg_to_set_lamda0, state_results, interpolants=None):
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

@dataclass
class TrajectoryAnalysis:
    integrand_terms: list[callable]
    terminal_terms: list[callable]

    def __call__(self, result):
        # evaluate the trajectory analysis of this result
        # should this return a dataclass? Or just the vector of results?
        pass


@dataclass
class ShootingGradientMethod:
    d_integrand_terms_d_params: list[callable]
    d_terminal_terms_d_params: list[callable]
    d_dots_d_params: callable
    d_update_d_params: list[callable]
    d_event_time_d_params: list[callable]
    d2_event_time_d_params_d_t: list[callable]

    def __call__(self, adjoint_result):
        pass


