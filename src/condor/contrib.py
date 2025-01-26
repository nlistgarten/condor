import logging

import casadi
import numpy as np

from condor.backends.casadi.utils import flatten
from condor.backends.default import backend
from condor.fields import (AssignedField, BaseElement, BoundedAssignmentField,
                           Direction, Field, FreeAssignedField, FreeElement,
                           FreeField, InitializedField, MatchedField,
                           TrajectoryOutputField, WithDefaultField)
from condor.models import (Model, ModelTemplate, ModelTemplateType, ModelType,
                           SubmodelTemplate)
import ndsplines

log = logging.getLogger(__name__)


class DeferredSystem(ModelTemplate):
    """output is an explicit function of input"""

    input = FreeField()
    output = FreeField(Direction.output)


# TODO: Move to "contrib" or something?
class ExplicitSystem(ModelTemplate):
    r"""output is an explicit function of input

    .. math::
       \begin{align}
       y_1 &=& f_1(x_1, x_2, \dots, x_n) \\
       y_2 &=& f_2(x_1, x_2, \dots, x_n) \\
       & \vdots & \\
       y_m &=& f_m(x_1, x_2, \dots, x_n)
       \end{align}

    where each :math:`y_i` is a name-assigned expression on the ``output`` field and each
    :math:`x_i` is an element drawn from the ``input`` field.

    Each :math:`x_i` and :math:`y_j` may have arbitrary shape. Condor can automatically
    calculate the derivatives :math:`\frac{dy_j}{dx_i}` as needed for parent solvers, etc.

    """

    input = FreeField() #: the inputs of the model (e.g., :math:`x_i`)
    output = AssignedField() #: the output of the model (e.g., :math:`y_i = f_i(x)`)


import casadi as ca


class AlgebraicSystemType(ModelType):
    @classmethod
    def process_placeholders(cls, new_cls, attrs):
        super().process_placeholders(new_cls, attrs)
        for elem in new_cls.residual:
            if elem.backend_repr.op() == ca.OP_EQ:
                lhs = elem.backend_repr.dep(0)
                rhs = elem.backend_repr.dep(1)
                elem.backend_repr = lhs - rhs


class AlgebraicSystem(ModelTemplate, model_metaclass=AlgebraicSystemType):
    r"""Represents a system of algebraic equations, with parameters 
    :math:`u` and implicit variables :math:`x`, which are driven to a solution
    :math:`x^*`:

    .. math::
       \begin{align}
       R_1(u_1, \dots, u_m, x_1^*, \dots, x_n^*) &=& 0 \\
       \vdots & & \\
       R_n(u_1, \dots, u_m, x_1^*, \dots, x_n^*) &=& 0
       \end{align}

    Condor solves for the :math:`x_i^*` and can automatically calculate the derivatives
    :math:`\frac{dx_i}{du_j}` as needed for parent solvers, etc.

    variable are variables that drive residual to 0
    parameters are additional variables parameterizing system's explicit_output and
    residuals
    output are additional fields that depend on (solved) value of variable and parameters

    uses AlgebraicSystem type as model metaclass to support relational declaration of
    residual.
    """
    parameter = FreeField() #:
    variable = InitializedField(Direction.output)
    residual = FreeAssignedField(Direction.internal)
    # unmatched, but maybe a subclass or imp might check lengths of residuals and
    # implicit_outputs to ensure enough DOF?
    output = AssignedField()

    def set_initial(cls, **kwargs):
        r"""set initial values for the ``variable``\s of the model.
        """
        for k, v in kwargs.items():
            var = getattr(cls, k)
            if var.field_type is not cls.variable:
                raise ValueError(
                    "Use set initial to set the initialier for variables, attempting to set {k}"
                )
            var.initializer = v


class OptimizationProblemType(ModelType):
    @classmethod
    def process_placeholders(cls, new_cls, attrs):
        super().process_placeholders(new_cls, attrs)
        original_elements = list(new_cls.constraint)
        new_cls.constraint._elements = []
        for elem in original_elements:
            # check if the backend_repr is a comparison op
            # check if the bounds are constant
            re_append_elem = True
            relational_op = False
            if elem.backend_repr.is_binary():
                lhs = elem.backend_repr.dep(0)
                rhs = elem.backend_repr.dep(1)

            if not isinstance(elem.lower_bound, np.ndarray) or np.any(
                np.isfinite(elem.lower_bound)
            ):
                real_lower_bound = True
            else:
                real_lower_bound = False

            if not isinstance(elem.upper_bound, np.ndarray) or np.any(
                np.isfinite(elem.upper_bound)
            ):
                real_upper_bound = True
            else:
                real_upper_bound = False

            if elem.backend_repr.op() in (ca.OP_LT, ca.OP_LE):
                relational_op = True
                elem.backend_repr = rhs - lhs
                elem.lower_bound = 0.0
            # elif elem.backend_repr.op() in (ca.OP_GT, ca.OP_GE):
            #    relational_op = True
            #    elem.backend_repr = rhs - lhs
            #    elem.upper_bound = 0.0
            elif elem.backend_repr.op() == ca.OP_EQ:
                relational_op = True
                elem.backend_repr = rhs - lhs
                elem.upper_bound = 0.0
                elem.lower_bound = 0.0

            if relational_op and (real_lower_bound or real_upper_bound):
                raise ValueError(
                    f"Do not use relational constraints with bounds for {elem}"
                )

            original_backend_repr = elem.backend_repr
            if real_lower_bound:
                if isinstance(elem.lower_bound, backend.symbol_class):
                    if not elem.lower_bound.is_constant():
                        # new_elem = elem.copy_to_field(new_cls.constraint)
                        new_elem = new_cls.constraint(
                            original_backend_repr - elem.lower_bound, lower_bound=0.0
                        )
                        re_append_elem = False
            if real_upper_bound:
                if isinstance(elem.upper_bound, backend.symbol_class):
                    if not elem.upper_bound.is_constant():
                        # new_elem = elem.copy_to_field(new_cls.constraint)
                        new_elem = new_cls.constraint(
                            original_backend_repr - elem.upper_bound, upper_bound=0.0
                        )
                        re_append_elem = False

            if re_append_elem:
                new_cls.constraint._elements.append(elem)

        # handle
        p = casadi.vertcat(*flatten(new_cls.parameter))
        x = casadi.vertcat(*flatten(new_cls.variable))
        g = casadi.vertcat(*flatten(new_cls.constraint))
        f = getattr(new_cls, "objective", 0.0)

        new_cls._meta.objective_func = casadi.Function(
            f"{new_cls.__name__}_objective",
            [x, p],
            [f],
        )
        new_cls._meta.constraint_func = casadi.Function(
            f"{new_cls.__name__}_constraint",
            [x, p],
            [g],
        )


class OptimizationProblem(ModelTemplate, model_metaclass=OptimizationProblemType):
    r"""Solve an optimization problem of the form

    .. math::

       \begin{aligned}
       \operatorname*{minimize}_{x_{1}, \ldots, x_{n}} &  &  &
            f (x_1, \ldots, x_n, p_1, \ldots p_m ) \\
       \text{subject to} &  &  & l_{x_i} \le x_i \le u_{x_i} \\
       & & & l_{g_i} \le g_i (x_1, \ldots , x_n, p_1, \ldots p_m ) \le u_{g_i} \\
       \end{aligned}


    variable are what the optimizer moves to solve problem, bounded and initialized
    parameter are fixed wrt optimization, but parameterize solution
    constraints get assigned as relationals

    user provides objective to minimize -- defaults to 0. (a feasibility problem)


    """
    variable = InitializedField(Direction.output)
    parameter = FreeField()
    # TODO: add objective descriptor? so user code `objetive = ...` can get intercepted and
    # handled?  or it's just a contract in the API that user writes one? feasibility
    # problem if not? Add hook for metaclass to handle it?
    # needs validation for size == 1
    constraint = BoundedAssignmentField(Direction.internal)
    objective = placeholder()

    def set_initial(cls, **kwargs):
        for k, v in kwargs.items():
            var = getattr(cls, k)
            if var.field_type is not cls.variable:
                raise ValueError(
                    "Use set initial to set the initialier for variables, attempting to set {k}"
                )
            var.initializer = v

    def from_values(cls, **kwargs):
        self = cls.__new__(cls)
        parameters = {}
        for elem in cls.parameter:
            val = kwargs.pop(elem.name)
            parameters[elem.name] = val
            setattr(self, elem.name, val)

        variables = {}
        for elem in cls.variable:
            val = kwargs.pop(elem.name)
            variables[elem.name] = val
            setattr(self, elem.name, val)

        if kwargs:
            raise ValueError(f"Extra arguments provided: {kwargs}")

        self.input_kwargs = parameters
        self.parameter = cls.parameter._dataclass(**parameters)
        self.variable = cls.variable._dataclass(**variables)

        args = [
            casadi.vertcat(*flatten(self.variable.asdict().values())),
            casadi.vertcat(*flatten(self.parameter.asdict().values())),
        ]
        constraints = cls._meta.constraint_func(*args)
        self.bind_field(cls.constraint.wrap(constraints))

        self.objective = cls._meta.objective_func(*args)

        self.bind_embedded_models()

        return self


class ODESystem(ModelTemplate):
    r""" Representation of a dynamical system defined by

    .. math::
       \begin{align}
       \dot{x}_i &= f(t,x_1,\ldots,x_n,p_1,\ldots,p_m) \\
       y_i &= h(t,x_1,\ldots,x_n,p_1,\ldots,p_m)
       \end{align}

    - ``modal`` - independent field, must always have an active ``action`` (or default)
    - ``state`` - independent field
    - ``dot`` - matched field, ``dot[x1] = ...``
    - ``initial`` - arguably belongs on TrajectoryAnalysis, not ODEsystem. matched field
      ``initial[x1] = ...``


    """
    """

    t - indepdendent variable of ODE, notionally time but can be used
    for anything. Used directly by subclasses (e.g., user code may use
    `u=DynamicsModel.t`, implementations will use this symbol
    directly for fields)

    parameter - auxilary variables (constant-in-time) that determine system behavior

    state - fully defines evolution of system driven by ODEs

    dot - derivative of state with respect to indepdnent variable, purely a function of
    state, independent_variable, parameters
    [DAE version will have a residual associated with state differential]

    output - dynamic output at particular state and value of independent variable
    (point-wise in independent variable)

    control - time-varying placeholder, useful for re-defining using mode, etc.
    set it with `make` field -- all controls MUST be set? or default to 0.

    make - set control a value/computation. Or `let`?

    note: no time-varying input to system. Assume dynamicsmodels "pull" what they need 
    from other models. 
    Need to augment state with "pulled" dynamicsmodels
    But do need a "control" that could be defined based on mode? automatically added to
    output? or is it always just a placeholder for functional space gradient?
    Or also allow it to define eg feedback control

    I guess block diagram is nice for simulating something like saturation block --
    create event and switch modes? but I guess that's re-creatable with modes. But maybe
    needs control/subsitution -- I guess what I'm calling "control" is really an
    (explicit) algebraic state? I guess this is really the same as an "output" and is in
    fact how simupy implements it. maybe convert output to a freefield that takes an
    expression (like constraint, I guess?) and then "make" is only in mode and adds
    conditional behavior -- defaults to value from creation?

    then keep control field but only for open loop control? not sure, but some way to
    mark an output as a control signal

    don't include specialty types like discrete control, etc? user must know that for
    discrete time signal it requires state with dot=0 and update

    dt - reserved keyword for DT?

    inner model "Event" (see below). For a MyODESyStem model, create a sub-class of
    MyODESystem.Event,
    inner classes of "Mode" subclass -- conditional overwrite of dot and make
    inner classes of TrajectoryAnalysis -- perform simulations with particular
    trajectory outputs. Local copy of parameters, initial to allow sim-specific



    """

    # TODO: indepdent var  needs its own descriptor type? OR do we want user classes to do
    # t = DynamicsModel.independent_variable ? That would allow leaving it like this and
    # consumers still know what it is
    # or just set it to t and always use it? trying this way...

    # TODO: Are initial conditions an attribute of the dynamicsmodel or the trajectory
    # analysis that consumes it?
    # Currently, ODESystem sets a default that TrajectoryAnalysis can override for
    # current sim. I think I like this

    # TODO: mode and corresponding FiniteState type?
    # Yes mode with condition, no FiniteState -- this is an idiom for a state with no
    # dot and only updated to integer values. Expect to use in conditions with ==. Can
    # python enum values get passed to backend?

    # TODO: convenience for setting max_step based on discrete time systems? And would
    # like event functions of form (t - te) to loop with a specified tf. For now can set
    # max_step in trajectoryanalysis options manually... OR figure out how to specify
    # periodic exact-time simulation and then current machinery should be fine

    # TODO: decide how controls work :) Ideally, easy to switch between continuous and
    # sampled feedback, open-loop control. Open-loop would need some type of
    # specification so adjoint gradients could get computed??

    # TODO: combine all events if functions are the same? maybe that happens
    # automatically and that's okay. Needs the new events API (mimicking sundials) to happen
    # --> VERIFY that SGM works correctly? Would require non-conflicting update
    # functions

    # TODO: inside another model, the callback gets captured into the "expression" and
    # will never hit the model's or implementation's __call__ methods. Would really like
    # to 1) bind the simulation result (as in the implementation call) and/or 2)
    # initialize the model since that is most likely where we would hook the
    # sweepable-like logging. This is also/really a question about how to inject Model
    # layer back into casadi expressions. Could be used for capturing dangling
    # computations which currently leaning away from doing but trajectory is a special
    # case... If the trajectory analysis only appears once in the other model could use
    # the caching of result object on callback to
    # Can set an attribute (`from_implementation`) on callback  assuming we make sure
    # new  instances of implementation (and callback) are created to ensure re-entrant.
    # Then refactor binding to create new model instance as appropriate. This would be
    # version #2, probably can't figure out what model called the trajectory analysis
    # and bind in which is probably more consistent anyway.

    # TODO: does the simulation result need to be included in the template somehow?
    # Or is this an implementation detail and current approach is fine? Or how to
    # indicate that time, state, and output fields are time varying and should be
    # written as an arrray and not to eg the database?

    # SimuPy-coupled TODO

    # TODO switch to scikits.odes, especially updating update function API to get an
    # array of length num_events w/ elements 0 for did not occur and +/-1 for direction
    # don't use nan's to terminate? althogh, can't code-gen termination?

    # --> MAKE SURE scikits.odes is re-entrant. AND important TODO: figure out how to
    # create new implementation instances as needed for parallelization. Not sure if
    # that would happen automatically with pickle, etc.

    # TODO: simupy currently requires dim_output > 0, which kinda makes sense for the
    # BlockDiagram case and kinda makes sense for disciplined dynamical system modeling,
    # but maybe doesn't make sense for single system simulation?
    # IF we were keeping intermediate computations, they would be good candidates

    # OR can/should trajectory output integrandds get added to simupy outputs, save a
    # for-loop?

    # what is a good output for adjoint system? gradient integrand term!

    # TODO: currently, to get a discrete time control need to augment state and provide
    # a separate initializer, even though it's  generally going to be the same
    # expression as the update. Should that be fixed in simupy? event funciton = 0 -> do
    # update? I can fix it in casadi shooting_gradient_method.py as well.

    # or is initial where we want it? consistent with initial conition processing for
    # SGM.

    # Are simupy outputs even good? need to be disciplined to keep system encapsolation
    # anyway, and control as discrete state prevents the re-computation. I guess
    # back-to-back computation of a "continuous" optimization (not sure if DAEs can do
    # this) is fine too? I think DAE version will be even more efficient, since
    # everything is a "state" and computed simultaneously. Can you do MAUD approach in
    # IDA to provide solvers for explicit expressions?

    # TODO: add event times/event channels to SimulationResult during detection

    # TODO: don't like hacks to simupy to make it work... especially the success
    # checking stuff -- is that neccessary anymore?

    # tf and t0 as placeholders with default 0 and inf, respectively
    # t = backend.symbol_generator('t') # TODO use placeholder with default = None
    t = placeholder(default=None)  # TODO use placeholder with default = None

    state = FreeField(Direction.internal)
    initial = MatchedField(state)
    parameter = FreeField()
    dot = MatchedField(state)
    modal = WithDefaultField(Direction.internal)
    dynamic_output = AssignedField(Direction.internal)


from dataclasses import dataclass, field

from condor.models import SubmodelMetaData, SubmodelType


@dataclass
class TrajectoryAnalysisMetaData(SubmodelMetaData):
    events: list = field(default_factory=list)
    modes: list = field(default_factory=list)


class TrajectoryAnalysisType(SubmodelType):
    """Handle kwargs for including/excluding events (also need to include/exlcude
    modes?), injecting bound events (event functions, updates) to model, etc.

    A common use case will be to bind the parameters then only update the state...


|                           |------------+-----------------------+----------------------|

|                           |            | constraint            |                      |
    """

    metadata_class = TrajectoryAnalysisMetaData

    @classmethod
    def __prepare__(
        cls,
        *args,
        include_events=None,
        exclude_events=None,
        include_modes=None,
        exclude_modes=None,
        **kwargs,
    ):
        cls_dict = super().__prepare__(*args, **kwargs)
        if exclude_events is not None and include_events is not None:
            raise ValueError("Use only one of include or include events")

        if include_events is None:
            cls_dict.meta.events = list(cls_dict.meta.primary.Event)
        else:
            cls_dict.meta.events = include_events

        if exclude_events is not None:
            cls_dict.meta.events = [
                event for event in cls_dict.meta.events if event not in exclude_events
            ]

        if exclude_modes is not None and include_modes is not None:
            raise ValueError("Use only one of include or include modes")

        if include_modes is None:
            cls_dict.meta.modes = list(cls_dict.meta.primary.Mode)
        else:
            cls_dict.meta.modes = include_modes

        if exclude_modes is not None:
            cls_dict.meta.modes = [
                mode for mode in cls_dict.meta.modes if mode not in exclude_modes
            ]

        return cls_dict

    def __new__(
        cls,
        *args,
        include_events=None,
        exclude_events=None,
        include_modes=None,
        exclude_modes=None,
        **kwargs,
    ):
        new_cls = super().__new__(cls, *args, **kwargs)
        return new_cls


class TrajectoryAnalysis(
    # ModelTemplate,
    SubmodelTemplate,
    model_metaclass=TrajectoryAnalysisType,
    primary=ODESystem,
    copy_fields=True,
):
    """"""

    """
    this is what simulates an ODE system
    tf parameter 
    define trajectory outputs
    modify parameters and initial (local copy)

    gets _res, t, state, output (from odesystem) assigned from simulation

    """
    trajectory_output = TrajectoryOutputField()
    tf = placeholder(default=np.inf)  # TODO use placeholder with default = None
    t0 = placeholder(default=0.0)  # TODO use placeholder with default = None

    # TODO: how to make trajectory outputs that depend on other state's outputs without
    # adding an accumulator state and adding the updates to each event? Maybe that
    # doesn't make sense...

    def point_analysis(cls, t, *args, **kwargs):
        self = ODESystem.__new__(ODESystem)
        """Compute the rates for the ODESystems that were bound (at the time of
        construction). Need equivalent for dynamic outputs, and ???
        """
        # bind paramaeters, state, call implementation functions (dot, dynamic output)

        # apply to dynamic output, as well? but needs the embedded models? hmm...
        # and I guess should similarly bind the events? -- a dictionary for update,
        # OR maybe dot/dynamic output is more simple, it's useful for trimming /other
        # solvers and doesn't need additional use case? there are potentially lots of
        # places embedded models came from -- I guess just events and modals?


# TODO: need to exlcude fields, particularly dot, initial, etc.
# define which fields get lifted completely, which become "read only" (can't generate
# new state) etc.
# maybe allow creation of new parameters (into ODESystem parameter field), access to
# state, etc.
class Event(
    # ModelTemplate,
    SubmodelTemplate,
    primary=ODESystem,
):
    """"""

    """
    update for any state that needs it

    terminate is a boolean flag to indicate whether the event terminates the simulation
    (default is False)

    function defines when event occurs OR at_time is an expression or slice of
    expressions for when the event occurs. Slice assumes constant dt between start and
    stop. use stop=None for infinite oscillator. Differs from standard slice
    semantics in that both start and stop are inclusive, so slice(0, 1, 1) will generatw
    two events at 0 and 1. This means an at_time of a single value of t_e is equiavelent
    to slice(t_e, t_e, None)


    How to bind numerical evaluation of model?
    During trajectory analysis model construction, currently existing events get copied
    to a local namespace and implementations are created, <trajectory analysis model
    name>.Events.<event model name>.
    I guess this is done by defining new  __new__ on TA, and calling super? Or do I need
    those hooks after all?
    Event implementation is owned by TA implementation, maybe in sub-name space to avoid
    extraneous implementation constructions.
    then when trajectory analysis is evaluated and bound, add events... somewhere. Maybe
    the _res.e elements get replaced by the evaluated Event models (so one for each
    occurance of the event)??? function isn't necessary, so maybe get an index instead.

    """
    # TODO: singleton field event.function is very similar to objective in
    # OptimizationProblem. And at_time. Need to be able to define such singleton
    # assignment fields by name so there's no clash for repeated symbols.
    update = MatchedField(ODESystem.state, direction=Direction.output)
    # make[mode_var] = SomeModeSubclass
    # actually, just update it
    # make = MatchedField(ODESystem.finite_state)
    # terminate = True -> return nan instead of update?

    terminate = placeholder(default=False)
    function = placeholder(default=np.nan)
    at_time = placeholder(default=np.nan)


# this should just provide the capabiility to overwrite make (or whatever sets control)
# and dot based on condition...
# needs to inject on creation? Or is TrajectoryAnalysis implementation expected to
# iterate Modes and inject? Then can add dot and make to copy_fields
class Mode(
    # ModelTemplate,
    SubmodelTemplate,
    primary=ODESystem,
):
    """"""

    """
    convenience for defining conditional behavior for state dynamics and/or controls
    depending on `condition`. No condition to over-write behavior, essentially a way to
    do inheritance for ODESystems which is otherwise hard? Can this be used instead of
    deferred subsystems? Yes but only for ODESystems..
    """
    condition = placeholder(default=1.0)
    action = MatchedField(ODESystem.modal, direction=Direction.internal)


def LTI(
    A,
    B=None,
    dt=0.0,
    dt_plant=False,
    name="LTISystem",
):
    attrs = ModelTemplateType.__prepare__(name, (ODESystem,))
    attrs["A"] = A
    state = attrs["state"]
    x = state(shape=A.shape[0])
    attrs["x"] = x
    xdot = A @ x
    if dt <= 0.0 and dt_plant:
        raise ValueError

    if B is not None:
        attrs["B"] = B
        K = attrs["parameter"](shape=B.T.shape)
        attrs["K"] = K

        if dt and not dt_plant:
            # sampled control
            u = state(shape=B.shape[1])
            attrs["u"] = u
            # attrs["initial"][u] = -K@x
        else:
            # feedback control matching system
            u = -K @ x
            attrs["dynamic_output"].u = u

        xdot += B @ u

    if not (dt_plant and dt):
        attrs["dot"][x] = xdot

    plant = ModelTemplateType(name, (ODESystem,), attrs)

    if dt:
        dt_attrs = SubmodelType.__prepare__("DT", (plant.Event,))
        # dt_attrs["function"] = np.sin(plant.t*np.pi/dt)
        dt_attrs["at_time"] = slice(None, None, dt)
        if dt_plant:
            from scipy.signal import cont2discrete

            if B is None:
                B = np.zeros((A.shape[0], 1))
            Ad, Bd, *_ = cont2discrete((A, B, None, None), dt=dt)
            dt_attrs["update"][dt_attrs["x"]] = (Ad - Bd @ K) @ x
        elif B is not None:
            dt_attrs["update"][dt_attrs["u"]] = -K @ x
            # dt_attrs["update"][dt_attrs["x"]] = x
        DTclass = SubmodelType(
            "DT",
            (plant.Event,),
            attrs=dt_attrs,
        )

    return plant


def copy_field(new_model_name, old_field, new_field=None):
    if new_field is None:
        new_field = old_field.inherit(new_model_name, field_type_name=old_field._name)
    new_field._elements = [sym for sym in old_field]
    return new_field


class ExternalSolverWrapperType(ModelTemplateType):
    """"""

    """
    since this is only one case, only need to bind input and output fields
    explicitly -- and possibly this is just syntax sugar. wrapper doesn't need
    much from the metclass, primarily handled by ExternalSolverModel which is
    automatically generated by wrapper's __create_model__, injected automatically
    by model sub-types. Could just as easily create decorator or other
    class-method (on Wrapper) that consumes singleton/collection of functions

    IO fields just a nice way to create simple object with IO metadata. 
    """

    @classmethod
    def __prepare__(cls, model_name, bases, name="", **kwds):
        log.debug(
            f"ExternalSolverWrapperType prepare for cls={cls}, model_name={model_name}, bases={bases}, name={name}, kwds={kwds}"
        )
        if name:
            model_name = name
        sup_dict = super().__prepare__(model_name, bases, **kwds)
        if cls.baseclass_for_inheritance is not None:
            if ExternalSolverWrapper in bases:  # should check MRO, I guess?
                # print("copying IO fields to", model_name)
                for field_name in ["input", "output"]:
                    sup_dict[field_name] = copy_field(
                        model_name, getattr(ExternalSolverWrapper, field_name)
                    )
        return sup_dict

    def __call__(cls, *args, **kwargs):
        log.debug(
            f"ExternalSolverWrapperType __call__ for cls={cls}, *args={args}, **kwargs={kwargs}"
        )
        # gets called on instantiation of the user wrapper, so COULD return the
        # condor model instead of the wrapper class -- perhaps this is more condoric,
        # not sure what's preferable
        # actully, this can get used instead of create model with init wrapper? no,
        # don't have access to instance yet.
        # print(cls, "__call__")
        wrapper_object = super().__call__(*args, **kwargs)
        return wrapper_object.condor_model


class ExternalSolverModel(ModelTemplate):
    input = FreeField()
    output = FreeField(Direction.output)


class ExternalSolverWrapper(
    ModelTemplate,
    metaclass=ExternalSolverWrapperType,
):
    input = FreeField()
    output = FreeField(Direction.output)

    def __init_subclass__(cls, singleton=True, **kwargs):
        log.debug(
            f"ExternalSolverWrapper init subclass, cls={cls}, singleton={singleton}, kwargs={kwargs}"
        )
        # at this point, fields  are already bound by ExternalSolverWrapperType.__new__
        # but modifications AFTER construction can be doen here
        # print("init subclass of", cls)
        cls.__original_init__ = cls.__init__
        cls.__init__ = cls.__create_model__

    def __create_model__(self, *args, condor_model_name="", **kwargs):
        log.debug(
            f"ExternalSolverWrapper create model self={self}, args={args}, condor_model_name={condor_model_name}, kwargs={kwargs}"
        )
        # print("create model of", self, self.__class__)
        # copy field so that any field modification by __original_init__ is onto the
        # copy
        # print("copying IO fields to", condor_model_name)
        if not condor_model_name:
            # could use repr somehow? but won't exist yet...
            condor_model_name = self.__class__.__name__
        for field_name in ["input", "output"]:
            setattr(
                self,
                field_name,
                copy_field(condor_model_name, getattr(self, field_name)),
            )

        self.__original_init__(*args, **kwargs)
        # update and/or copy meta? -- no, create a __condor_model__ class which is the
        # actual model and call, etc. get mapped to that??

        attrs = ExternalSolverModel.__prepare__(
            condor_model_name, (ExternalSolverModel,)
        )
        # this copying feels slightly redundant...
        for field_name in ["input", "output"]:
            copy_field(
                condor_model_name,
                getattr(self, field_name),
                new_field=attrs[field_name],
            )

        self.condor_model = ExternalSolverModel.__class__(
            condor_model_name, (ExternalSolverModel,), attrs
        )
        self.condor_model._meta.external_wrapper = self


class TableLookup(ExternalSolverWrapper):
    """The output is the interpolated value for each input """

    # TODO enforce shape = 1 for input/output??
    """
    rectilnear -> structured, rectilinear grid
    later, unstructured lookup.

    inputs define input variables
    outputs are names of output

    knots are vector of grid points for each input
    data is grid of data for each output of shape generated by meshgrid ij of all knots

    detect if knots and/or data are symbolic or numeric. If symbolic, treat as input

    options:
    free or fixed?
    spline degree

    """
    def __init__(self, xx, yy, degrees=3, bcs=(-1,0)):
        input_data = []
        for k, v in xx.items():
            self.input(name=k)
            input_data.append(v)
        output_data = []
        for k, v in yy.items():
            self.output(name=k)
            output_data.append(v)
        output_data = np.stack(output_data, axis=-1)
        self.interpolant = ndsplines.make_interp_spline(
            input_data, output_data, degrees=degrees, bcs=bcs,
        )
        self.jac_interps = [
            self.interpolant.derivative(idx) for idx in range(self.interpolant.xdim)
        ]
        self.hess_interps = [
            [
                interpolant.derivative(idx)
                if interpolant.degrees[idx] > 0
                else lambda *args: np.zeros((1,interpolant.ydim))
                for idx in range(interpolant.xdim)
            ]
            for interpolant in self.jac_interps
        ]

    def function(self, xx):
        return self.interpolant(np.array(xx).reshape(-1))[0, :]#.T

    def jacobian(self, xx):
        array_vals = [
            interp(np.array(xx).reshape(-1))[0, :]
            for interp in self.jac_interps
        ]
        # TODO -- original implementation did not have transpose, but generic version
        # needs it
        # EVEN WORSE, adding hessian capability makes it want to have transpose again??
        # some weird casadi issue I assume... :(
        # changing API of casadi's FunctionToOperator to return the value (and letting
        # casadi-specific do casadi-specific thing) means don't transpose?
        return_val = np.stack(array_vals, axis=1)
        return return_val

    def hessian(self, xx):
        array_vals = np.stack([
            np.stack([
                interp(np.array(xx).reshape(-1))[0, :]
                for interp in interp_row
            ], axis=0)
            for interp_row in  self.hess_interps
        ], axis=1)
        return array_vals

