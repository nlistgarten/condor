class BaseLinCovCW:
    t = independent_variable() # maybe this is in the base class, can get over-written?

    # parameters defined in the class, hopefully accessible by scope (but maybe not
    # without scope magic)
    measurement_frequency, measurement_phase = parameters()

    # metaclass can definitely process the unpacking? or does unpacking automatically
    # create sequentially keyed items so it would just work to return 1?
    # so could make that feel really fun

    omega = constant(...) # define it as a constant? or... just set a value? lol

    # can figure out how to manage scope for parameters. stack overflow probably follows
    # similar mechanism for unpacking

    # for MDAO, parameters imply inputs?
    # 
    # demo GASPy with control flow type states (t_rotate)
    # demo orbital, iterator to create major/minor burns - 
    # how to specify state equations? maybe as a dict?? state generation defines order
    # special dict attribute state_rate[state_var] = ...
    # in this context, "output" is of trajectory?
    # could always correspond to integral of lagrange term (cost to go?) -- augmented
    # state with state rate = L (integrand)
    # can create state rate from overloading simupy but can't inherently inspect
    # interior signals, only IO which is very systems-y. May need symbolic vector to
    # dispatch operations? neat.
    # neeed to automate passing up of metadata from a simupy bd -- 
    # similar system output as attribute dict with state keysa, ?? o
    # can't have computational blocks?? unless they provide a derivative?
    # need to create own symbolic representation 



    # numerical representation: ~ simupy syntax
    # I guess not really needed since the whole point is to handle all the needed
    # partial derivatives
    dim_state = ...
    dim_output = ...
    dim_input = ...

    def state_equation(t, x, u):
        return ...

    # symbolic representation 1
    state = ...
    input = ...
    output = ...

    state_equation = ...


    # name the event by subclassing
    class measurement(event):
        # event.function is the event function?
        function = sin(measurement_frequency*t + measurement_phase)


        # need some symbol for returning back to "same" phase,
        # is it always the most specific subclass? probably.
        # then need a way to terminate?
        # maybe we assume there is never a "next phase" everything defined by events?
        # how to handle GASPy? 
        # just create new state, 
        next_phase = cyclic





class LinCovCW(SGM):
    t = independent_variable() # maybe this is in the base class, can get over-written?

    # in future work this maybe optimized -- not sure if there will be an API for
    # parameters we aren't optimizing over? 
    measurement_frequency, measurement_phase = constant(..., ...)

    # call state() to generate state variables. I think we can handle unpacking
    px, py, pz = state()
    vx, vy, vz = state()

    # state_rate[:] defines all currently defined states? can index like a numpy array
    # or by list of state references
    # similarly, state without a call gets all currently defined state?
    state_rate[...] = A_cw @ state
    AugCov = state(12, 12)
    state_rate[AugCov] = Acal @ AugCov + AugCov @ Acal + ...

    NavCov = state(6, 6)

    mu = state() # Delta v nominal accumulator
    sigma = state() # Delta v dispersion accumulator

    state_rate[mu] = 0. # actually a default, so don't need to do this/

    initial[state] = ....

    class measurement(Event):
        # event.function is the event function?
        function = sin(measurement_frequency*t + measurement_phase)

        # default/unset behavior for update is continuous
        update[NavCov] = ...

    class BurnTemplate(EventTemplate):
        tig = parameter_template() # don't need template suffix since declared in
        # template?
        tmf = parameter_template()
        burn_target = parameter_template(3)

        function = t - tig
        update[] = ...

    for maj_burn_idx, num_minor_burns = enumerate([0, 2, 5, 3]):

        # everything that needs to get automatically indexed gets indexed?
        # parameter with repeat names, event classes, etc.
        # parameters declared here won't have a persistent accessor by name on main
        # class; instead access throguh class.parameters

        # I guess parameters need initial guesses?
        tig_maj = parameter()
        tmf_maj = parameter()
        burn_target_maj = parameter(3)


        class MajorBurn(BurnTemplate, tig=tig_maj, tmf=tmf_maj, burn_target_maj):
            pass


        for min_burn_idx in range(num_minor_burns):
            tig_min = parameter()


            class MinorBurn(MajorBurn, tig=tig_min):
                pass



        # or...
        MajorBurn = BurnTemplate(tig=tig_maj, tmf=tmf_maj, burn_target_maj)
        for min_burn_idx in range(num_minor_burns):
            tig_min = parameter()
            MinorBurn = MajorBurn(tig=tig_min)




class AircraftMission(SGM):
    # control is ~ time-varying parameter
    # state is time-varying
    # output is 
    alpha = control()
    throttle = control()
    weight, drange, altitude, gamma, airspeed  = state()


    mode = finite_state()
    # logic can only depend on finite_state, mode can only be updated by events
    # can only 


    # these states will automatically not matter since they don't directly appear in
    # output integrand of terminal terms?
    t_rot, t_gear, t_flaps = state()

    initial[t_rot, t_gear, t_flaps] = inf

    # basically provides a convenience to using symbolic logic operators (casadi
    # if_else, sympy piecewise, etc). symbolic logic operators in backend are fine
    # I guess it's just doing piecewise for you for the stated expressions?

    class groundroll(mode):
        alpha = 0.

    initial[mode] = groundroll

    class rotation(mode):
        alpha = rot_rate * (t - rot)

    class rotation_trigger(Event):
        function = airspeed - v_rot
        # time and state on RHS of update means at event time
        update[t_rot] = t
        update[mode] = rotation

    class rotating_ascent(rotation):
        pass

    class 

    class constrained_ascent(mode):
        alpha = max(
            solve(TAS_constraint),
            solve(xlf_constraint),
            solve(pitch_constraint)
        )





    # similar events for gear and flap retraction time
    # alpha for rotation, aero for gear/flap retraction are easy with this formulation

    # want fully determined controls? or allow functional space deriv? no, can put
    # spline with finite parameters unless study shows its worse?

    # generally expressions are point-wise in time
    fuel_flow_rate, thrust = propulsion(flight_condition, throttle)





class trajectory_analysis(model=some_ode_class,):
    class output1(output):
        integrand_cost = expression(model.state, control, etc)
        terminal_cost = expressino(...)

    input = ....
    model.p = input
    # so trajectory analysis distinguishes between parameters (inputs?) and sets
    # constants, not the ODE!!




