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
    #function = t - terminate_time
    at_time = terminate_time,

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

    #function = ca.sin(np.pi*(t-meas_t_offset)/meas_dt)
    #function = meas_dt*ca.sin(np.pi*(t-meas_t_offset)/meas_dt)/np.pi
    #function = t - meas_t_offset
    at_time = (meas_t_offset, None, meas_dt)


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
    meas_t_offset = 1.
)

scenario_1_target = [-200., 0., 0.]
scenario_1_kwargs = dict(
    #  (0 km; −10 km; 0.2 km; 0 m∕s; 0 m∕s; 0 m∕s)
    # The initial relative states (radial, along-track, and cross-track) are 
    # x0 (0 km; −10 km; 0.2 km; 0 m∕s; 0 m∕s; 0 m∕s), and the desired final relative
    # states are xf (0 km;−0.2 km;0 km;0 m∕s;0 m∕s;0 m∕s)

    # CW frame in meters:
    # x is down-track with -x behind, +x in front
    # y is cross-track,
    # z altitude with +z below, -z above
    initial_x=[-10_000., 200, 0., 0., 0., 0.,],
)

scenario_2_target = [200., 0., 0.]
scenario_2_kwargs = dict(
    #The initial relative states (radial, along-track, and cross- track) are x0
    # (1 km; 10 km; 0.2 km; 0 m∕s; −1.7 m∕s; 0.2 m∕s, and the desired final relative 
    # states are xf (0 km; 0.2 km; 0 km; 0 m∕s; 0 m∕s; 0 m∕s). 
    initial_x=[10_000., 200., 1_000., 1.71, 0., 0.,],
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

scenario_a_tf = 2000.
first_tig = 0. # TODO: handle this.... 
scenario_1a = [
    Sim(
        **base_kwargs,
        **scenario_1_kwargs,
        **cost_system_kwargs,
        terminate_time=scenario_a_tf,
        tem_1=scenario_a_tf,
        tig_1=first_tig,
        rd_1=scenario_1_target,
    )
    for cost_system_kwargs in [low_cost_kwargs, nominal_kwargs, high_cost_kwargs]
]
scenario_2a = [
    Sim(
        **base_kwargs,
        **scenario_2_kwargs,
        **cost_system_kwargs,
        terminate_time=scenario_a_tf,
        tem_1=scenario_a_tf,
        tig_1=first_tig,
        rd_1=scenario_2_target,
    )
    for cost_system_kwargs in [low_cost_kwargs, nominal_kwargs, high_cost_kwargs]
]


print([sim.final_pos_disp for sim in scenario_1a])
print([sim.final_vel_disp + sim.Delta_v_disp_1 for sim in scenario_1a])
print([sim.final_vel_mag + sim.Delta_v_mag_1 for sim in scenario_1a])

print([sim.final_pos_disp for sim in scenario_2a])
print([sim.final_vel_disp + sim.Delta_v_disp_1 for sim in scenario_2a])
print([sim.final_vel_mag + sim.Delta_v_mag_1 for sim in scenario_2a])

# the dispersions are the same between scenario 1a and 2a, which makes sense -- just
# propagating covariance between the first burn and final burn. 1b and 2b look like they
# introduce more variations I assume because the intermediate burns are doing something
# with linear CW dynamics 1a and 2a are identical, paper's nonlinearity tweaks it
# slightly but not much



MajorBurn = make_burn(
    rd = LinCovCW.parameter(shape=3), # desired position
    tig = LinCovCW.parameter(), # time ignition
    tem = LinCovCW.parameter(), # time end maneuver
)
Sim = make_sim()
scenario_kwargs = dict(
    **scenario_1_kwargs,
    terminate_time=scenario_a_tf,
)
scenario_target = scenario_1_target
cost_system_kwargs = low_cost_kwargs


class Deterministic2Burn1a(co.OptimizationProblem):
    rds = []
    n_burns = len(make_burn.burns)
    burn_config = dict(tig_1=first_tig)
    ts = [first_tig]
    for burn_num, burn, next_burn in zip(range(1,n_burns+2), make_burn.burns, make_burn.burns[1:]):
        ratio = burn_num/n_burns
        ts.append(variable(
            name=f"t_{burn_num+1}",
            initializer=scenario_kwargs['terminate_time']*ratio
        ))
        rds.append(variable(
            name=f"pos_{burn_num+1}",
            shape=(3,),
            initializer=np.array(scenario_kwargs["initial_x"][:3])*(1-ratio)+np.array(scenario_target)*ratio
        ))
        constraint(ts[-1]-ts[-2], lower_bound=10.)
        burn_config[LinCovCW.parameter.get(backend_repr=burn.tem).name] = ts[-1]
        burn_config[LinCovCW.parameter.get(backend_repr=next_burn.tig).name] = ts[-1]
        burn_config[LinCovCW.parameter.get(backend_repr=burn.rd).name] = rds[-1]
    constraint(ts[-1], upper_bound=scenario_kwargs['terminate_time'])
    burn_config[LinCovCW.parameter.get(backend_repr=next_burn.tem).name] = scenario_kwargs['terminate_time']
    burn_config[LinCovCW.parameter.get(backend_repr=next_burn.rd).name] = scenario_target

    sim = Sim(
        **base_kwargs,
        **scenario_kwargs,
        **cost_system_kwargs,
        **burn_config
    )
    objective = sim.final_vel_mag + 3*sim.final_vel_disp
    for burn_num in range(n_burns):
        objective += getattr(sim, f"Delta_v_mag_{burn_num+1}") + 3*getattr(sim, f"Delta_v_disp_{burn_num+1}")

    class Casadi(co.Options):
        exact_hessian=False
        #method = OptimizationProblem.Method.scipy_trust_constr



import sys
sys.exit()
opt = Deterministic2Burn1a()
"""
WITH SIMUPY VERSION

trying to do deterministic optimization with extra  burn in middle got stuck on
[-5076.54, 6.88871, 2582.95, 0.1, 1021.07, -200, 0, 0, 1021.07, 2000]
rd: -5076.54, 6.88871, 2582.95,
tig: 1021.07

initialzied with:
rd: array([-5.1e+03,  1.0e-01,  0.0e+00])
tig: 1000.


optimal trajectory for 1a with 1 burn:
In [27]: x[:3, 0]
Out[27]: array([-1.e+04,  2.e-01,  0.e+00])

In [24]: np.where(t>1000.)[0]
Out[24]: array([1322, 1323, 1324, ..., 2620, 2621, 2622])
In [25]: x[:3, 1322]
Out[25]: array([-5.09753611e+03,  2.34991856e-01,  2.58765334e+03])

In [16]: np.where(t>1021.)
Out[16]: (array([1350, 1351, 1352, ..., 2620, 2621, 2622]),)
In [23]: x[:3, 1350]
Out[23]: array([-4.95464320e+03,  2.32351375e-01,  2.58638861e+03])

In [26]: x[:3, -1]
Out[26]: array([-2.00000000e+02,  2.25210475e-15,  1.60198868e-08])


so i5 had actually moved the altitude coordinate z to be aligned with the trajectory
which isn't at all linear in the coordinate system which is neat! 

also, the last iteration in ipopt was
  96  9.4742132e+00 0.00e+00 1.70e-03 -11.0 6.54e+01    -  1.00e+00 4.44e-16f 52
which seems close to the minimum value

In [30]: scenario_1a[0].Delta_v_mag_1 + scenario_1a[0].final_vel_mag
Out[30]: 9.474213195999624

so quite good. Since there are at least a 1DOF family of solutions (for any tig there is
at least 1 rd that is effectively a null burn) it makes sense the optimizaiton got
stuck. is ipopt just jumping around the curve? probably.

would scipy's SLSQP do any better? since it's an illconditioned problem, problaby not.
The only other interesting thing would be to have a different burn model with Delta v
directly under optimization control. This is similar to the SOCP model I wrote. This may
de-couple the variables enough the optimizer can find that tig doesn't matter if Delta v
-> 0. And the 1-burn case because less constrained (although, could let the final rd be
a variable as well and it should be able to work in either case.


with new version,
Deterministic2Burn1AVariable(
t_2=694.5854379512559,
pos_2=array([[-7.15808799e+03],
       [ 2.58572260e-01],
       [ 2.32135487e+03]]))


nom = scenario_1a[1]._res
x = nom.x
t = np.array(nom.t)
x[np.where((t > 690) & (t < 700))[0][0]][:6]
array([-7.12781868e+03,  2.58437104e-01,  2.32952679e+03,  6.35514913e+00,
       -2.91236881e-05,  1.70027993e+00])

so again optimizer is just moving around (very closely!) on optimal trajectory which is
sweet. I possibly let it go longer, and it might have been moving up the trajectory
which I think makes sense for decreasing cost (but again, it's singular)...


"""

scenario_b_tf = 12_000.
scenario_kwargs = dict(
    **scenario_1_kwargs,
    terminate_time=scenario_b_tf,
)


MajorBurn = make_burn(
    rd = LinCovCW.parameter(shape=3), # desired position
    tig = LinCovCW.parameter(), # time ignition
    tem = LinCovCW.parameter(), # time end maneuver
)
Sim = make_sim()

class Deterministic3Burn1b(co.OptimizationProblem):
    rds = []
    n_burns = len(make_burn.burns)
    burn_config = dict(tig_1=first_tig)
    ts = [first_tig]
    for burn_num, burn, next_burn in zip(range(1,n_burns+2), make_burn.burns, make_burn.burns[1:]):
        ratio = burn_num/n_burns
        ts.append(variable(
            name=f"t_{burn_num+1}",
            initializer=scenario_kwargs['terminate_time']*ratio
        ))
        rds.append(variable(
            name=f"pos_{burn_num+1}",
            shape=(3,),
            initializer=np.array(scenario_kwargs["initial_x"][:3])*(1-ratio)+np.array(scenario_target)*ratio
        ))
        constraint(ts[-1]-ts[-2], lower_bound=10.)
        burn_config[LinCovCW.parameter.get(backend_repr=burn.tem).name] = ts[-1]
        burn_config[LinCovCW.parameter.get(backend_repr=next_burn.tig).name] = ts[-1]
        burn_config[LinCovCW.parameter.get(backend_repr=burn.rd).name] = rds[-1]
    constraint(ts[-1], upper_bound=scenario_kwargs['terminate_time'])
    burn_config[LinCovCW.parameter.get(backend_repr=next_burn.tem).name] = scenario_kwargs['terminate_time']
    burn_config[LinCovCW.parameter.get(backend_repr=next_burn.rd).name] = scenario_target

    sim = Sim(
        **base_kwargs,
        **scenario_kwargs,
        **cost_system_kwargs,
        **burn_config
    )
    objective = sim.final_vel_mag + 3*sim.final_vel_disp
    for burn_num in range(n_burns):
        objective += getattr(sim, f"Delta_v_mag_{burn_num+1}") + 3*getattr(sim, f"Delta_v_disp_{burn_num+1}")

    class Casadi(co.Options):
        exact_hessian=False
        #method = OptimizationProblem.Method.scipy_trust_constr

opt2 = Deterministic2Burn1b()

#opt = Burn1(sigma_Dv_weight=3, mag_Dv_weight=1, sigma_r_weight=0)
#opt_sim = Sim(
#        tig_1=opt.t1,
#        **sim_kwargs
#)

print("\n"*3,"burn time minimization")
print(opt._stats)

