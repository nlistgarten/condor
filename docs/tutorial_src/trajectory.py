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
# angle-of-attack control.
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
# lift, :math:`CL`, and drag, :math:`CD`.  Simple models of the lift and drag are given
# by
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
from condor import operators as ops
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


# %%
# The :attr:`modal` field is used to define elements with deferred and possibly varying
# behavior, so we use this for the angle-of-attack so we can simulate multiple
# behaviors. To simulate this model, we create a :class:`TrajectoryAnalysis`, a sub-model
# to an ODE Sysstem, which is ultimately responsible for defining the specifics of
# integrating the ODE. In this example, we will fix the initial conditions and final
# simulation time of the model.

class FirstSim(Glider.TrajectoryAnalysis):
    initial[r] = 0.
    initial[h] = 1.
    initial[v] = 15.
    initial[gamma] = 30*ops.pi/180.
    tf = 20.

# %%
# The fields of the original :class:`Glider` simulation are copied to the
# :class:`TrajectoryAnalysis` so the parameter values must be supplied to evaluate the
# model numerically.

A = 3E-1
first_sim = FirstSim(CL_alpha = 2*ops.pi*A, CD_0 = 0.05*A, CD_i_q = 0.05, g=1.)

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

def flight_path_plot(sims):
    fig, ax = plt.subplots(constrained_layout=True)
    plt.ylabel("altitude")
    plt.xlabel("range")
    for idx, sim in enumerate(sims):
        ax.plot(sim.r, sim.h, f'C{len(sims)-idx-1}o-')
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
# :class:`FirstSim` was bound to the :class:`Glider` model at the time of creation.

class BounceSim(Glider.TrajectoryAnalysis):
    initial[r] = 0.
    initial[h] = 1.
    initial[v] = 15.
    initial[gamma] = 30*ops.pi/180.
    tf = 20.

bounce_sim = BounceSim(**first_sim.parameter.asdict(), mu=0.9)

flight_path_plot([bounce_sim, first_sim])

# %%
# and we can also add a behavior for the angle of attack using a mode, in this case
# holding a constant half-degree angle of attack after reaching peak altitude to reduce
# rate of descent.

class AlphaAttack(Glider.Mode):
    condition = gamma < 0
    action[alpha] = 0.5*ops.pi/180.

class MaxAlt(Glider.Event):
    function = gamma

class AlphaSim(Glider.TrajectoryAnalysis):
    initial[r] = 0.
    initial[h] = 1.
    initial[v] = 15.
    initial[gamma] = 30*ops.pi/180.
    tf = 20.

alpha_sim = AlphaSim(**bounce_sim.parameter.asdict())

flight_path_plot([alpha_sim, bounce_sim, first_sim])

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
