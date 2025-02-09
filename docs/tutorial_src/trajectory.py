"""
=========================
Working with trajectories
=========================
"""
# %%
# Condor is not intended to be an optimal control library per-se, but we often end up
# working with trajectories a great deal and spent considerable effort to make modeling
# dynamical systems nice.
#
# For this tutorial, we will consider a simplified model of a glider with some form of
# angle-of-attack control. We can represent this as a system of ordinary differential
# equations (ODEs) given by
#
# .. math::
#    \begin{align}
#    \dot{r} &= v \cos \gamma \\
#    \dot{h} &= v \sin \gamma \\
#    \dot{\gamma} &= (CL(\alpha) \cdot v^2 - g \cos \gamma) / v \\
#    \dot{v} &= - CD(\alpha) \cdot v^2 - g \sin \gamma \\
#    \end{align}
#
# where :math:`r` is the range, or horizontal position, :math:`h` is the altitude, or
# vertical position, :math:`v` is the velocity,  :math:`\gamma` is the flight-path
# angle, and :math:`\alpha` is the angle-of-attack, which modulates the coefficients of
# lift, :math:`CL`, and drag, :math:`CD`, and :math:`g` is the acceleration due to
# gravity. Simple models of the lift and drag are given by
#
# .. math::
#    \begin{align}
#    CL(\alpha) &= CL_{\alpha} \cdot \alpha \\
#    CD(\alpha) &= CD_0 + CD_{i,q} \cdot CL^2 \\
#    \end{align}
#
# where :math:`CL_{\alpha}` is the lift slope, :math:`CD_0` is the 0-lift drag, and
# :math:`CD_{i,q}` is the quadratic coefficient for the lift-induced drag. In Condor, we
# can implement this as,

import condor
from condor.backend import operators as ops
class Glider(condor.ODESystem):
    r = state()
    h = state()
    gamma = state()
    v = state()

    alpha = modal()

    CL_alpha = parameter()
    CD_0 = parameter()
    CD_i_q = parameter()
    g = parameter()

    CL = CL_alpha * alpha
    CD = CD_0 + CD_i_q * CL**2


    dot[r] = v * ops.cos(gamma)
    dot[h] = v * ops.sin(gamma)
    dot[gamma] = (CL * v**2 - g*ops.cos(gamma) )/v
    dot[v] = -CD * v**2 - g*ops.sin(gamma)

    initial[r] = 0.
    initial[h] = 1.
    initial[v] = 15.
    initial[gamma] = 30*ops.pi/180.


# %%
# The :attr:`modal` field is used to define elements with deferred and possibly varying
# behavior, so we use this for the angle-of-attack so we can simulate multiple
# behaviors. To simulate this model, we create a :class:`TrajectoryAnalysis`, a sub-model
# to an ODE Sysstem, which is ultimately responsible for defining the specifics of
# integrating the ODE. In this example, the :class:`TrajectoryAnalysis` model only
# specifies the final simulation time of the model.  It is more mathematically
# consistent to have the initial values defined in the trajectory analysis, but for
# convience we declared it as part of the ODE system.

class FirstSim(Glider.TrajectoryAnalysis):
    tf = 20.

# %%
# The fields of the original :class:`Glider` simulation are copied to the
# :class:`TrajectoryAnalysis` so the parameter values must be supplied to evaluate the
# model numerically.
#

A = 3E-1
first_sim = FirstSim(CL_alpha = 0.11*A, CD_0 = 0.05*A, CD_i_q = 0.05, g=1.)

# %%
# In addition to binding static parameters like the other built-in models, the
# time-histories for the :attr:`state` and :attr:`dynamic_output` are bound and can be
# accessed for plotting. For example, we can plot time histories

from matplotlib import pyplot as plt
def time_history_plot(sims):
    fig, axs = plt.subplots(nrows=4, constrained_layout=True, sharex=True)
    axs[0].set_ylabel('altitude')
    axs[1].set_ylabel('range')
    axs[2].set_ylabel('flight path angle (deg)')
    axs[3].set_ylabel('velocity')

    for sim in sims:
        axs[0].plot(sim.t, sim.h.T)
        axs[1].plot(sim.t, sim.r.T)
        axs[2].plot(sim.t, sim.gamma.T*180/ops.pi)
        axs[3].plot(sim.t, sim.v.T)

    for ax in axs:
        ax.grid(True)
    ax.set_xlabel('time')

time_history_plot([first_sim])

# %%
# and the flight path

def flight_path_plot(sims, line_spec="o-"):
    fig, ax = plt.subplots(constrained_layout=True)
    plt.ylabel("altitude")
    plt.xlabel("range")
    for idx, sim in enumerate(sims):
        ax.plot(sim.r, sim.h, f'C{len(sims)-idx-1}{line_spec}')
    plt.grid(True)
    ax.set_aspect('equal')
    plt.ylim(0., 30)
    plt.xlim(0, 80)

flight_path_plot([first_sim])

# %%
# We can demonstrate the use of an :class:`Event` to model the bounce off the ground, 

class Bounce(Glider.Event):
    function = h
    update[gamma] = -gamma
    mu = parameter()
    update[v] = mu*v

# %%
# but we still need to create a new :class:`TrajectoryAnalysis` since the
# :class:`FirstSim` was bound to the :class:`Glider` model at the time of creation
# (without the bounce event).

class BounceSim(Glider.TrajectoryAnalysis):
    tf = 20.

bounce_sim = BounceSim(**first_sim.parameter.asdict(), mu=0.9)
first_sim = FirstSim(CL_alpha = 0.11*A, CD_0 = 0.05*A, CD_i_q = 0.05, g=1.)

flight_path_plot([bounce_sim, first_sim])
plt.legend(["with bounce", "original sim"])

# %%
# We can also add a behavior for the angle of attack using a mode, in this case
# holding a constant half-degree angle of attack after reaching peak altitude to reduce
# rate of descent. To ensure proper numerical behavior, we follow [orbital ref] and use 
# of an accumulator state to encode the flight controller logic. In this case we track
# the range at the start of descent, and use that as a trigger for holding a non-zero
# angle of attack.

class MaxAlt(Glider.Event):
    function = gamma
    #accum_state = state(name="range_at_max_alt")
    #initial[accum_state] = 1E3
    #update[accum_state] = r * (dot[r] > 0)
    accum_state = state(name="max_alt")
    initial[accum_state] = 0
    update[accum_state] = h * (h > accum_state) + accum_state * (h <= accum_state)


class AlphaAttack(Glider.Mode):
    condition = gamma < 0
    condition = (r >= MaxAlt.accum_state)# * (gamma < 0)
    condition = 1.
    hold_alpha = parameter()
    action[alpha] = hold_alpha

class AlphaSim(Glider.TrajectoryAnalysis):
    initial[r] = 0.
    initial[h] = 1.
    initial[v] = 15.
    initial[gamma] = 30*ops.pi/180.
    tf = 20.

alpha_sim = AlphaSim(**bounce_sim.parameter.asdict(), hold_alpha = 0.5)

flight_path_plot([alpha_sim, bounce_sim, first_sim])


# %%
# So far, we have only used the :class:`TrajectoryAnalysis` to simulate the ODE System.
# In order to use the ODE system as part of other condor models, we must declare
# :attr:`trajectory_output`. Condor computes the gradient of the
# :attr:`trajectory_output` using the Sweeping Gradient method. Each trajectory output
# has the form
#
# .. math::
#    J = \phi\left(t_{f},x\left(t_{f}\right),p\right)+\int_{t_{0}}^{t_{f}}L\left(\tau,x\left(\tau\right),p\right)\,d\tau
#
# where :math:`\phi` is the terminal term, :math:`L\left(\cdot\right)` is the integrand
# term, and :math:`x\left(t\right)` is the solution to the system of ODEs with events
# and modes. We can form the area under the flight-path curve by taking the derivative
# and using it to form the integrand.

Bounce.terminate = True

class AlphaSim(Glider.TrajectoryAnalysis):
    initial[r] = 0.
    initial[h] = 1.
    initial[v] = 15.
    initial[gamma] = 30*ops.pi/180.
    tf = 100.

    area = trajectory_output(integrand=dot[r]*h)
    max_h = trajectory_output(MaxAlt.accum_state)
    max_r = trajectory_output(r)
    #area = trajectory_output(integrated_area)
    #area = trajectory_output(r)
    #area = trajectory_output(-t)
    #area2 = trajectory_output(integrand=h)

    class Options:
        if True:
            state_rtol = 1E-12
            state_atol = 1E-15
            adjoint_rtol = 1E-12
            adjoint_atol = 1E-15
        #state_solver = condor.backend.implementations.TrajectoryAnalysis.Solver.dop853





# %%
# Then we can compare areas,

results = {
    "alpha = +0.5 deg": AlphaSim(**bounce_sim.parameter.asdict(), hold_alpha = 0.5),
    "alpha = 0.0 deg": AlphaSim(**bounce_sim.parameter.asdict(), hold_alpha = 0.0),
    "alpha = -0.5 deg": AlphaSim(**bounce_sim.parameter.asdict(), hold_alpha = -0.5),
}

print(*[f"{k}: {v.area}" for k, v in results.items()])
#print(*[f"{k}: {v.area2}" for k, v in results.items()])

flight_path_plot(results.values())
plt.legend([k.replace("alpha", r"$\alpha$") for k in results])


# %%
# With several :attr:`trajectory_output`, we can embed it within other Condor models,
# for example to maximize a combination of the  area under the curve and maximum
# altitude. We will also use the :attr:`iter_callback` to save each trajectory the
# optimizer tries. This code block will declare the optimization problem, run it, and
# plot each trajectory.

class Callback:
    def __init__(self):
        self.parameter = None

    @classmethod
    def attach(cls, options):
        options._callback = cls()
        options.init_callback = options._callback.init_callback
        options.iter_callback = options._callback.iter_callback

    def init_callback(self, parameter, impl_opts):
        self.parameter = parameter
        print("starting optimization with...")
        #print(impl_opts)
        print(parameter)
        self.sims = []

    def iter_callback(self, i, variable, objective, constraint):
        iter_opt_res = MaxArea.from_values(
            **variable.asdict(),
            **self.parameter.asdict(),
        )
        self.sims.append(iter_opt_res)
        print(f"iter {i}: {variable.asdict()}, area={iter_opt_res.sim.area}")


class MaxArea(condor.OptimizationProblem):
    alpha = variable(
        initializer=0.001,
        lower_bound=-1.,
        upper_bound=1,
        warm_start=False,
    )
    sim = AlphaSim(**bounce_sim.parameter.asdict(), hold_alpha = alpha)
    trade_off = parameter()
    objective = -(trade_off*sim.area + (1-trade_off)*sim.max_h)
    objective = -(trade_off*sim.area + (1-trade_off)*sim.max_r)
    objective = -(trade_off*sim.max_h + (1-trade_off)*sim.max_r)

    class Options:
        exact_hessian = False
        print_level = 0
        tol = 1E-3
        max_iter = 8


Callback.attach(MaxArea.Options)
for title, to in {
        #"max altitude": 0.,
        #"max area": 1.,
        "max range": 0.,
        "max altitude": 1.,
}.items():
    print("\n"*3, title)
    opt = MaxArea(trade_off=to)
    opt_sims = opt.Options._callback.sims
    flight_path_plot([sim.sim for sim in opt_sims][::-1], '-')
    plt.legend([f"iter {idx}, alpha={sim.alpha}" for idx, sim in enumerate(opt_sims)][::-1])
    plt.title(f"optimization iterations for {title}")
    plt.ylim(0, 45)
    plt.xlim(0, 85)

plt.show()

# %%
# ..
#    # For this tutorial, we will consider a simple two-dimensional double integrator with an
#    # acceleration input,
#    #
#    # ..math::
#    #    \begin{align}
#    #    \dot{p} &= v \\
#    #    \dot{v} &= a
#    #    \end{align}
#    #
#    # where :math:`p` is the position, :math:`v` is the velocity, and :math:`a` is the
#    # control velocity, all in two-dimensions. In Condor, 
#
#    import condor
#    class DblInt(condor.ODESystem):
#        p = state(shape=2)
#        v = state(shape=2)
#
#        a = modal(shape=2)
#
#        dot[p] = v
#        dot[v] = a
#
#
#    class BasicSim(DblInt.TrajectoryAnalysis):
#        initial_pos = parameter(shape=2)
#        initial_vel = parameter(shape=2)
#        initial[p] = initial_pos
#        initial[v] = hjkk
#
