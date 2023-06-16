import numpy as np
import condor as co
import casadi as ca
import matplotlib.pyplot as plt

from condor.backends.casadi.implementations import OptimizationProblem
from LinCovCW import LinCovCW, make_burn, I3, Z3, I6, Z6, deriv_check_plots, make_sim


class Terminate(LinCovCW.Event):
    terminate = True
    terminate_time = parameter()
    # TODO: how to make a symbol like this just provide the backend repr? or is this
    # correct?
    function = t - terminate_time

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
    M = ca.vertcat( ca.horzcat(I6, Z6), ca.horzcat(M01, Acalhat) )
    N = ca.vertcat(Z3, Z3, Khat)
    update[C] = M @ C @ M.T + N @ Rcal @ N.T

    #update[Delta_v_disp] = Delta_v_disp# + ca.sqrt(sigma_Dv__2)
    #update[Delta_v_mag] = Delta_v_mag# + ca.sqrt(sigma_Dv__2)
    #update[x] = x

    meas_dt = parameter()
    meas_t_offset = parameter()

    function = ca.sin(np.pi*(t-meas_t_offset)/meas_dt)
    function = meas_dt*ca.sin(np.pi*(t-meas_t_offset)/meas_dt)/np.pi
    #function = t - meas_t_offset


def add_measurement_params(kwargs):
    kwargs.update(dict(
        scalhat_w=kwargs['scal_w'],
        scalhat_v=kwargs['scal_v'],
        rcalhat=kwargs['rcal'],
        initial_P=kwargs['initial_C'][-6:, -6:] - kwargs['initial_C'][:6, :6],
    ))

def make_C0(pos_disp, vel_disp, nav_pos_err=None, nav_vel_err=None):
    if nav_pos_err is None:
        nav_pos_err = pos_disp
    if nav_vel_err is None:
        nav_vel_err = vel_disp
    D0 = np.diag(np.r_[pos_disp, vel_disp])
    P0 = np.diag(np.r_[nav_pos_err, nav_vel_err])
    return np.block([[D0, D0], [D0, D0+P0]])
"""
recreating Geller_RobustTrajectoryDesign

variance for actuator error  Q_w  eq (33)
variance for measurement noise R_nu eq (37)

To simplify the forthcoming analysis, process noise has been removed from the filter
and the true dynamics. This is done merely to reduce the number of problem parameters
and can be easily added to the problem formulation in the future.



"""
mu_earth = 3.986_004e14
r_earth_km = 6_378.1
alt_km = 400.
r_orbit_m = (r_earth_km+alt_km)*1E3

base_kwargs = dict(
    omega = np.sqrt(mu_earth/((r_orbit_m)**3)),
    meas_dt = 10.,
    meas_t_offset = 0.
)

scenario_1_kwargs = dict(
    #  (0 km; −10 km; 0.2 km; 0 m∕s; 0 m∕s; 0 m∕s)
    # The initial relative states (radial, along-track, and cross-track) are 
    # x0 (0 km; −10 km; 0.2 km; 0 m∕s; 0 m∕s; 0 m∕s), and the desired final relative
    # states are xf (0 km;−0.2 km;0 km;0 m∕s;0 m∕s;0 m∕s)

    # CW frame in meters:
    # x is down-track with -x behind, +x in front
    # y is cross-track,
    # z altitude with +z below, -z above
    initial_x=[-10_000., 0.2, 0., 0., 0., 0.,],
    rd_1=[-200., 0., 0.],
)

scenario_2_kwargs = dict(
    #The initial relative states (radial, along-track, and cross- track) are x0
    # (1 km; 10 km; 0.2 km; 0 m∕s; −1.7 m∕s; 0.2 m∕s, and the desired final relative 
    # states are xf (0 km; 0.2 km; 0 km; 0 m∕s; 0 m∕s; 0 m∕s). 
    initial_x=[-2000., 0., 1000., 1.71, 0., 0.,],
    rd=[200., 0., 0.],
)

low_cost_kwargs = dict(
    scal_w=[0.]*6,
    initial_C=make_C0([1000.]*3, [1]*3),
    # measurement error
    rcal=[10.]*3,
    # control variance
    scal_v=[0.1]*3,
)
add_measurement_params(low_cost_kwargs)


nominal_kwargs = dict(
    scal_w=[0.]*6,
    initial_C=make_C0([100.]*3, [0.1]*3),
    # measurement error
    rcal=[1.]*3,
    # control variance
    scal_v=[0.01]*3,
)
add_measurement_params(nominal_kwargs)


high_cost_kwargs = dict(
    scal_w=[0.]*6,
    initial_C=make_C0([10.]*3, [.01]*3),
    # measurement error
    rcal=[.1]*3,
    # control variance
    scal_v=[0.001]*3,
)
add_measurement_params(high_cost_kwargs)



MajorBurn = make_burn(
    rd = LinCovCW.parameter(shape=3), # desired position
    tig = LinCovCW.parameter(), # time ignition
    tem = LinCovCW.parameter(), # time end maneuver
)

# 1-burn sim
Sim = make_sim()

scenario_1a = [
    Sim(
        **base_kwargs,
        **scenario_1_kwargs,
        **cost_system_kwargs,
        terminate_time=2000.,
        tig_1=0.1,
        tem_1=2000.,
    )
    for cost_system_kwargs in [low_cost_kwargs, nominal_kwargs, high_cost_kwargs]
]

[sim.final_pos_disp for sim in scenario_1a]
[sim.final_vel_disp + sim.Delta_v_disp_1 for sim in scenario_1a]
[sim.final_vel_mag + sim.Delta_v_mag_1 for sim in scenario_1a]


class Burn1(co.OptimizationProblem):
    t1 = variable(initializer=1900.)
    sigma_r_weight = parameter()
    sigma_Dv_weight = parameter()
    mag_Dv_weight = parameter()
    sim = Sim(
        tig_1=t1,
        **sim_kwargs

    )

    constraint(t1, lower_bound=30., upper_bound=sim_kwargs['tem_1']-30.)
    constraint(sim.final_pos_disp, upper_bound=10.)
    objective = (
        sigma_Dv_weight*sim.tot_Delta_v_disp
        + sigma_r_weight*sim.final_pos_disp
        + mag_Dv_weight*sim.tot_Delta_v_mag
    )
    class Casadi(co.Options):
        exact_hessian=False
        method = OptimizationProblem.Method.scipy_trust_constr

#opt = Burn1(sigma_Dv_weight=3, mag_Dv_weight=1, sigma_r_weight=0)
#opt_sim = Sim(
#        tig_1=opt.t1,
#        **sim_kwargs
#)

print("\n"*3,"burn time minimization")
print(opt._stats)

