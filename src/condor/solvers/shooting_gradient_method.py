import numpy as np
#from condor import backend
from scipy.interpolate import make_interp_spline

from dataclasses import dataclass, field, InitVar
from scikits.odes.sundials.cvode import CVODE, StatusEnum
from typing import NamedTuple

class Root(NamedTuple):
    index: int
    rootsfound: list[int]

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
        self, dim_state, initial_state, dot, jac, time_generator,
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

        self.dim_state = dim_state
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
        return np.array(self._initial_state(self.result.p)).squeeze()

    def dots(self, t, x, xdot,):# userdata=None,):
        xdot[:] = np.array(self._dot(self.result.p, t, x)).squeeze()

    def jac(self, t, x, xdot, jac):#
        jac[...] = np.array(self._jac(self.result.p, t, x)).squeeze()

    def events(self, t, x, g):
        g[:] = np.array(self._events(self.result.p, t, x)).squeeze()

    def update(self, t, x, rootsfound):
        next_x = x #np.copy(x) # who is responsible for copying? I suppose simulate
        for root_sign, update in zip(rootsfound, self._updates):
            if root_sign != 0:
                next_x = update(self.result.p, t, next_x)
        return np.array(next_x).squeeze()

    def time_generator(self):
        for t in self._time_generator(self.result.p):
            yield np.array(t).reshape(-1)[0]


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

        events have index corresponding to start of each segment. except perhaps the
        first. -- can always assume the 0 index corresponds to initialization
        initial and final segments may be singular if an event causes an update that
        coincides with 
        """
        results = self.result
        solver = self.solver
        last_x = self.initial_state()
        results.x.append(last_x)

        time_generator = self.time_generator()
        last_t = next(time_generator)
        results.t.append(last_t)

        gs = np.empty(self.num_events)
        self.events(last_t, last_x, gs)
        rootsfound = (gs == 0.).astype(int)
        if np.any(rootsfound):
            # subsequent events use length of time for index, so root index is the index
            # of the updated state to the right of event. -> root coinciding with 
            # initialization has index 1
            last_x = self.update(last_t, np.copy(last_x), rootsfound)
        results.e.append(Root(1, rootsfound))
        results.t.append(last_t)
        results.x.append(last_x)


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
                solver_res = solver.step(next_t)
                if solver_res.flag < 0:
                    breakpoint()

                results.t.append(np.copy(solver_res.values.t))
                results.x.append(np.copy(solver_res.values.y))

                if solver_res.flag == StatusEnum.ROOT_RETURN:
                    idx = len(results.t)
                    rootsfound = solver.rootinfo()
                    results.e.append(Root(idx, rootsfound))

                    next_x = self.update(
                        results.t[-1], results.x[-1], rootsfound,
                    )
                    terminate = np.any(rootsfound[self.terminating] != 0)
                    results.t.append(np.copy(solver_res.values.t))
                    results.x.append(next_x)

                    if terminate:
                        return

                    solver.init_step(res.values.t, next_x)
                    last_x = next_x

                if solver_res.values.t == next_t or solver_res.flag == StatusEnum.TSTOP_RETURN:
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

    def __getitem__(self, key):
        return self.__class__(
            *tuple(getattr(self, field.name) for field in fields(self)[:-3]),
            t=self.t[key], x=self.x[key], e=self.e[key]
        )



@dataclass
class Result(ResultBase, ResultMixin):
    pass

class ResultSegmentInterpolant(NamedTuple):
    interpolant: callable
    idx0: int
    idx1: int
    t0: float
    t1: float
    x0: list[float]
    x1: list[float]

    def __call__(self, t):
        return self.interpolant(t)

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

            self.event_idxs = np.array([
                root.index for root in result.e
            ])# + [len(result.t)])
            if np.any(np.diff(self.event_idxs) < 1):
                breakpoint()

        event_idxs = self.event_idxs
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
        # TODO figure out how to combine (and possibly reverse direction) state and
        # parameter jacobian of state equation to reduce number of calls (and distance
        # at each call), since this is potentially most expensive call -- I guess only
        # if the ODE depends on inner-loop solver? then split
        # coefficients
        # --> then also combine e.g., adjoint forcing function and jacobian interpolant

        # expect integrand-term related functions to be cheap functions of state
        # point-wise along trajectory

            self.interpolants = [
                ResultSegmentInterpolant(
                    make_interp_spline(
                        result.t[idx0:idx1],
                        [function( result.p, t, x,) 
                         for t, x in zip(result.t[idx0:idx1], result.x[idx0:idx1])],
                        k=min(3, idx1-idx0)
                    ),
                    idx0, idx1,
                    result.t[idx0], result.t[idx1],
                    result.x[idx0], result.x[idx1],
                )
                for idx0, idx1 in zip(
                    event_idxs[:-1],
                    event_idxs[1:]-1,
                )
                if result.t[idx1] != result.t[idx0]
            ]

    def __call__(self, t):
        interval_idx = np.where(self.time_comparison(t))[0][self.interval_select]
        return self.interpolants[interval_idx](t)

    def __iter__(self):
        for interp_segment in self.interpolants:
            yield interp_segment



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

        adjoint solver will use parent simulate but without using events machinery
        time generator will call update method
        """
        self.state_jac = state_jac
        self.num_events = 0

    def update(self, segment_idx, event):
        """
        for adjoint system, update will always get called for t1 of each segment, 
        """
        lamda_res = self.result
        state_res = lamda_res.state_result
        # might be able to some magic to skip if nothing is done? but might only be for
        # terminal event anyway? maybe initial?
        active_update_idxs = np.where(event.rootsfound != 0)[0]
        lamda_tep = last_lamda = self.result.x[-1]
        p = state_res.p


        if len(active_update_idxs) > 1:
            # not sure how to handle actual computation for this case, may need to
            # re-compute each update to get x at each update for correct partials and
            # time derivatives??
            breakpoint()


        if active_update_idxs:
            idxp = event.index # positive side of event
            te = state_res.t[idxp]
            xtep = state_res.x[idxp]
            ftep = state_res.system._dot(p, te, xtep)

            idxm = idxp -1
            if state_res.t[idxm] != te:
                breakpoint()
            xtem = state_res.x[idxm]
            ftem = state_res.system._dot(p, te, xtem)

            delta_fs = ftep - ftem
            delta_xs = xtep - xtem
            lamda_dot = np.empty(xtep.shape)
            self.dots(te, last_lamda, lamda_dot)

            for event_channel in active_update_idxs[::-1]:
                lamda_tem = (
                    self.dh_dxs[event_channel](p, te, xtem,).T
                    - self.dte_dxs[event_channel](p, te, xtem).T @ delta_fs.T
                    + self.d2te_dxdts[event_channel](p, te, xtem).T @ delta_xs.T
                ) @ last_lamda
                - 1*(
                     lamda_dot[None, :] @ (delta_xs) @ self.dte_dxs[event_channel](p, tem, xtem)
                ).T

                last_lamda = lamda_tem


        if event is result.state_result.e[-1]: # i.e., is terminal:
            # simulate already added an extra step as a matter of course, so this should
            # replace those values
            lamda_res.e[-1].rootsfound = event.rootsfound
            lamda_res.x[-1] = lamda_tem
        else:
            # append event, time, and state
            lamda_res.e.append(Root(len(lamda_res.t), event.rootsfound))
            lamda_res.t.append(te)
            lamda_res.x.append(lamda_tem)

            pass

    def time_generator(self):
        """
        """
        result = self.result
        for (
            segment_idx,
            event,
            #jac_segment, forcing_segment,
        ) in zip(
            range(len(result.state_result.e)-1, -1, -1),
            result.state_result.e[::-1],
            # result.state_jacobian[::-1], result.forcing_function[::-1],
        ):
            # the  event corresponds to the t0+ of the segment index which can be used
            # for selecting the jacobian and forcing segments
            self.segment_idx = segment_idx
            yield result.state_result.t[event.index]
            # nothing about segment_idx will get used the first time (terminal event)
            # and would be out-of-bounds if if tried -- simulate will use the yielded
            # time to determine inital condition, then calls next to set endpoint of
            # next segment then propoagate it.

            # so on first iteration, will not propoagate, will hit first iteration of
            # simulate's loop and call next-yield. so do terminal update (can optimize
            # code to reduce computations if needed) then loop.
            self.update(event)

            # so update for terminal event is right, but could optimize performance for
            # special cases/maybe need distinct expression for update (to implement
            # update from from true terminal condition to effect of an immediate update)
            # then when this yields initial event (i.e., index=1) will THEN propoagate
            # backwards to initial condition. 

        # then exits above loop, will have just simulated to t0 and handled any possible
        # update
        yield np.inf


    def initial_state(self):
        return self.final_lamda

    def __call__(self, final_lamda, forcing_function, state_jacobian):
        self.result = AdjointResult(
            system=self,
            state_jacobian=state_jacobian,
            forcing_function=forcing_function
        )
        self.final_lamda = final_lamda 
        # I guess this could be put in result.x? then initial_state can pop and return?
        self.simulate()
        return super().simulate()

    def dots(self, t, x, xdot,):# userdata=None,):
        # TODO adjoint system takes jacobian and integrand terms, do matrix multiply and
        # vector add here. then can ust call jacobian term for jac method
        # then before simulate, could actually construct 
        # who would own the data for interpolating ? maybe just a new (data) class that
        # also stores the interpolant? then adjointsystem simulate can create it if
        # it'snot provided
        xdot[:] = (
            -self.result.state_jacobian.interpolants[self.segment_idx](t).T @ x 
            -self.result.forcing_function.interpolants[self.segment_idx](t)
        )

    def jac(self, t, x, xdot, jac, userdata=None,):
        jac[...] = -self.result.state_jacobian.interpolants[self.segment_idx](t).T

@dataclass
class TrajectoryAnalysis:
    integrand_terms: callable
    terminal_terms: callable

    def __call__(self, result):
        # evaluate the trajectory analysis of this result
        # should this return a dataclass? Or just the vector of results?
        integral = 0.
        integrand_interpolant = ResultInterpolant(result=result, function=self.integrand_terms)
        for segment in integrand_interpolant:
            integrand_antideriv = segment.interpolant.antiderivative()
            integral += integrand_antideriv(segment.t1) - integrand_antideriv(segment.t0)
        return self.terminal_terms(result.p, result.t[-1], result.x[-1]) + integral


@dataclass
class ShootingGradientMethod:
    # per system
    adjoint_system: AdjointSystem
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


    def __call__(self, state_result):
        # iterate over each output, generate forcing functions, etc

        # TODO figure out combining and splitting to make this more efficient
        state_jacobian = ResultInterpolant(state_result, self.adjoint_system.state_jac)
        param_jacobian = ResultInterpolant(state_result, self.p_dots_p_params)

        p = state_result.p



        for (
            p_integrand_term_p_params,
            p_terminal_term_p_params,
            p_integrand_term_p_state,
            p_terminal_term_p_state,
        ) in zip(
            self.p_integrand_terms_p_params,
            self.p_terminal_terms_p_params,
            self.p_integrand_terms_p_state,
            self.p_terminal_terms_p_state,
        ):
            adjoint_forcing = ResultInterpolant(state_result, p_integrand_term_p_state)
            final_lamda = p_terminal_term_p_state(
                p, state_result.t[-1], state_result.x[-1]
            )
            adjoint_result = self.adjoint_system(
                final_lamda, adjoint_forcing, state_jacobian
            )

            adjoint_interp = ResultInterpolant(adjoint_result)

            for 

            #    simulate adjoint for each one
            #    iterate over each segment/event for both  state + adjoint
            #       compute discontinuous portion of gradient associated with each event
            #       integrate continuous portion of gradient corresponding to pre/suc-ceding
            #       segment
            pass


