import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from LinCovCW import (
    I3,
    I6,
    Z3,
    Z6,
    LinCovCW,
    deriv_check_plots,
    make_burn,
    make_sim,
    sim_kwargs,
)

import condor as co
from condor.backends.casadi.implementations import OptimizationProblem

MajorBurn = make_burn(
    rd=LinCovCW.parameter(shape=3),  # desired position
    tig=LinCovCW.parameter(),  # time ignition
    tem=LinCovCW.parameter(),  # time end maneuver
)


class Terminate(LinCovCW.Event):
    terminate = True
    # TODO: how to make a symbol like this just provide the backend repr? or is this
    # correct?
    # function = t - MajorBurn.tem
    at_time = [
        MajorBurn.tem,
    ]


class Measurements(LinCovCW.Event):
    rcal = parameter(shape=3)
    rcalhat = parameter(shape=3)

    Rcal = ca.diag(rcal)
    Rcalhat = ca.diag(rcalhat)

    Hcal = ca.horzcat(I3, Z3)
    Hcalhat = ca.horzcat(I3, Z3)

    Khat = (P @ Hcalhat.T) @ ca.solve(Hcalhat @ P @ Hcalhat.T + Rcalhat, ca.MX.eye(3))
    Acalhat = I6 - Khat @ Hcalhat
    update[P] = Acalhat @ P @ Acalhat.T + Khat @ Rcalhat @ Khat.T

    M01 = Khat @ Hcal
    M = ca.vertcat(ca.horzcat(I6, Z6), ca.horzcat(M01, Acalhat))
    N = ca.vertcat(Z3, Z3, Khat)
    update[C] = M @ C @ M.T + N @ Rcal @ N.T

    # update[Delta_v_disp] = Delta_v_disp# + ca.sqrt(sigma_Dv__2)
    # update[Delta_v_mag] = Delta_v_mag# + ca.sqrt(sigma_Dv__2)
    # update[x] = x

    meas_dt = parameter()
    meas_t_offset = parameter()

    # function = ca.sin(np.pi*(t-meas_t_offset)/meas_dt)
    # this re-normalizes the derivative to be ~1 at the 0-crossings which may or may not
    # be helpful
    # function = meas_dt*ca.sin(np.pi*(t-meas_t_offset)/meas_dt)/np.pi
    # function = t - meas_t_offset
    # at_time = [meas_t_offset, None, meas_dt]
    at_time = [
        meas_t_offset,
    ]


sim_kwargs.update(
    dict(
        meas_dt=2300.0,
        meas_t_offset=851.0,
        # meas_dt = 100.,
        # meas_t_offset = 51.,
    )
)

# 1-burn sim
# class Sim(LinCovCW.TrajectoryAnalysis):
#    # TODO: add final burn Delta v (assume final relative v is 0, can get magnitude and
#    # dispersion)
#    tot_Delta_v_mag = trajectory_output(
#        sum([burn.Delta_v_mag for burn in make_burn.burns])
#    )
#    #tot_Delta_v_disp = trajectory_output(Delta_v_disp)
#    tot_Delta_v_disp = trajectory_output(
#        sum([burn.Delta_v_disp for burn in make_burn.burns])
#    )
#
#    Mr = ca.horzcat(I3, ca.MX(3,9))
#    sigma_r__2 = ca.trace(Mr @ C @ Mr.T)
#    final_pos_disp = trajectory_output(ca.sqrt(sigma_r__2))
#
#    class Casadi(co.Options):
#        integrator_options = dict(
#            max_step = 1.,
#            #atol = 1E-15,
#            #rtol = 1E-12,
#            nsteps = 10000,
#        )
# Sim.implementation.callback.get_jacobian(
#     f"jac_{Sim.implementation.callback.name}", None, None, {}
# )


Sim = make_sim()
output_vars = Sim.Delta_v_mag_1, Sim.Delta_v_disp_1, Sim.final_pos_disp
sim_kwargs.update(
    dict(
        tem_1=3800.0,
        # meas_dt = 200.,
        # meas_t_offset = 7.5,
        # meas_t_offset = 851.,
    )
)


sim_kwargs.pop("meas_t_offset", None)
meas_times = np.arange(300, 1100, 5.0)
meas_times = np.arange(300, 900, 5.0)
meas_times = np.arange(0.0, 150.0, 20.0)
meas_times = np.r_[1:151:20, 153:203:20]
meas_sims = [
    (
        Sim(tig_1=152.0, meas_t_offset=meas_time, **sim_kwargs),
        Sim.implementation.callback.jac_callback(Sim.implementation.callback.p, []),
    )
    for meas_time in meas_times
]
indep_var = Sim.meas_t_offset
xx, yy = deriv_check_plots(Sim.meas_t_offset, output_vars, meas_sims, title_prefix="mt")


# plt.show()
# import sys
# sys.exit()

tigs = np.arange(600, 1000.0, 50)
# tigs = np.arange(10, 3000., 5)
tig_sims = [
    (
        Sim(tig_1=tig, meas_t_offset=851.0, **sim_kwargs),
        Sim.implementation.callback.jac_callback(Sim.implementation.callback.p, []),
    )
    for tig in tigs
]
indep_var = Sim.tig_1


deriv_check_plots(Sim.tig_1, output_vars, tig_sims, title_prefix="tig")

# plt.show()
# import sys
# sys.exit()


# 2-burn sim
sim_kwargs.update(
    dict(
        tem_1=1210.0,
        tig_1=10.0,
    )
)
MinorBurn = make_burn(
    rd=MajorBurn.rd,  # desired position
    tig=LinCovCW.parameter(),  # time ignition
    tem=MajorBurn.tem,  # time end maneuver
)

# class Sim2(LinCovCW.TrajectoryAnalysis):
#    # TODO: add final burn Delta v (assume final relative v is 0, can get magnitude and
#    # dispersion)
#    tot_Delta_v_mag = trajectory_output(Delta_v_mag)
#    tot_Delta_v_disp = trajectory_output(Delta_v_disp)
#
#    Mr = ca.horzcat(I3, ca.MX(3,9))
#    sigma_r__2 = ca.trace(Mr @ C @ Mr.T)
#    final_pos_disp = trajectory_output(ca.sqrt(sigma_r__2))

# Sim2.implementation.callback.get_jacobian(
#     f"jac_{Sim2.implementation.callback.name}", None, None, {}
# )
Sim2 = make_sim()

tigs = np.arange(600, 1000.0, 50)
sims = [
    (
        Sim2(tig_2=tig, meas_t_offset=1500.0, **sim_kwargs),
        Sim2.implementation.callback.jac_callback(Sim2.implementation.callback.p, []),
    )
    for tig in tigs
]
sims2 = sims
deriv_check_plots(Sim2.tig_2, output_vars, sims2, title_prefix="mcc tig")
"""
turning debug_level to 0 for shooting_gradient_method moves from 200s to 190s.

"""
plt.show()

import sys  # noqa

sys.exit()


# was attempting to optimize measurement time given fixed burn schedule: converse of
# orbital_3
class Meas1(co.OptimizationProblem):
    t1 = variable(initializer=100.0)
    sigma_r_weight = parameter()
    sigma_Dv_weight = parameter()
    mag_Dv_weight = parameter()
    sim = Sim(**sim_kwargs, meas_t_offset=t1, tig_1=900.0)

    constraint(t1, lower_bound=30.0, upper_bound=sim_kwargs["tem_1"] - 30.0)
    objective = (
        sigma_Dv_weight * sim.tot_Delta_v_disp
        + sigma_r_weight * sim.final_pos_disp
        + mag_Dv_weight * sim.tot_Delta_v_mag
    )

    class Casadi(co.Options):
        exact_hessian = False
        method = OptimizationProblem.Method.scipy_trust_constr


# opt = Meas1(sigma_Dv_weight=0, mag_Dv_weight=1, sigma_r_weight=1)
# sim = Sim(**sim_kwargs, meas_t_offset=100.)

fig, axes = plt.subplots(3, constrained_layout=True, sharex=True)
for ax, ordinate in zip(axes, ordinates):
    ax.plot(tigs, ordinate)
    ax.grid(True)


Dv_mags = [sim.tot_Delta_v_mag for sim in sims]
Dv_disps = [sim.tot_Delta_v_disp for sim in sims]
pos_disps = [sim.final_pos_disp for sim in sims]
ordinates = [Dv_mags, Dv_disps, pos_disps]

fig, axes = plt.subplots(3, constrained_layout=True, sharex=True)
for ax, ordinate in zip(axes, ordinates):
    ax.plot(tigs, ordinate)
    ax.grid(True)


plt.show()


# sim1 = Sim(
#        tem_1=opt.tf,
#        tig_1=850,
#        **sim_kwargs
# )
#
#
# sim2 = Sim(
#        tem_1=opt.tf,
#        tig_1=852,
#        **sim_kwargs
# )
