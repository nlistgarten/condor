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
scenario_b_tf = 12_000.
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


# the dispersions are the same between scenario 1a and 2a, which makes sense -- just
# propagating covariance between the first burn and final burn. 1b and 2b look like they
# introduce more variations I assume because the intermediate burns are doing something
# with linear CW dynamics 1a and 2a are identical, paper's nonlinearity tweaks it
# slightly but not much


####################
# DEFINE THE SCENARIO
####################

scenario_kwargs = dict(
    **scenario_1_kwargs,
    terminate_time=scenario_b_tf,
)
scenario_target = scenario_1_target
cost_system_kwargs = nominal_kwargs

# generate additional burns
for idx in range(2): 
    MajorBurn = make_burn(
        rd = LinCovCW.parameter(shape=3), # desired position
        tig = LinCovCW.parameter(), # time ignition
        tem = LinCovCW.parameter(), # time end maneuver
    )

Sim = make_sim()


class OptimizeBurns(co.OptimizationProblem):
    disp_weighting = parameter()
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
    objective = sim.final_vel_mag + disp_weighting*sim.final_vel_disp
    for burn_num in range(n_burns):
        objective += getattr(sim, f"Delta_v_mag_{burn_num+1}") + disp_weighting*getattr(sim, f"Delta_v_disp_{burn_num+1}")

    class Casadi(co.Options):
        exact_hessian=False
        #method = OptimizationProblem.Method.scipy_trust_constr

##########
# print basic results
#########

print([sim.final_pos_disp for sim in scenario_1a])
print([sim.final_vel_disp + sim.Delta_v_disp_1 for sim in scenario_1a])
print([sim.final_vel_mag + sim.Delta_v_mag_1 for sim in scenario_1a])

print([sim.final_pos_disp for sim in scenario_2a])
print([sim.final_vel_disp + sim.Delta_v_disp_1 for sim in scenario_2a])
print([sim.final_vel_mag + sim.Delta_v_mag_1 for sim in scenario_2a])


import sys
sys.exit()
opt = OptimizeBurns(disp_weighting=0.)
"""
analysis with "2 burn" -- I think this is actually 3 burns, because I wasn't counting
station keeping?
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

with 3 burns, I think trust_constr eventually worked? got an objective value similar to
ipopt but actually closed
In [4]: opt2.variable
Out[4]: 
Deterministic3Burn1AVariable(t_2=735.3199620174034, pos_2=array([[-6896.48148127],
       [  257.20017037],
       [ 2387.2343194 ]]), t_3=1257.515572589859, pos_3=array([[-3350.07016741],
       [  193.4188517 ],
       [ 2397.86145275]]))

In [5]: opt2.objective
Out[5]: 18.563590917695457

In [6]: opt2._stats
Out[6]: 
           message: `xtol` termination condition is satisfied.
           success: True
            status: 2
               fun: 18.563590917695457
                 x: [ 7.353e+02 -6.896e+03  2.572e+02  2.387e+03  1.258e+03
                     -3.350e+03  1.934e+02  2.398e+03]
               nit: 407
              nfev: 403
              njev: 403
              nhev: 0
          cg_niter: 1054
      cg_stop_cond: 2
              grad: [-2.428e-03  7.880e-04  1.923e-03 -1.018e-03 -1.407e-03
                     -4.959e-04 -1.296e-03 -2.436e-03]
   lagrangian_grad: [-2.428e-03  7.880e-04  1.923e-03 -1.018e-03 -1.407e-03
                     -4.959e-04 -1.296e-03 -2.436e-03]
            constr: [array([ 7.353e+02,  5.222e+02,  1.258e+03])]
               jac: [<3x8 sparse matrix of type '<class 'numpy.float64'>'
                    	with 4 stored elements in Compressed Sparse Row format>]
       constr_nfev: [0]
       constr_njev: [0]
       constr_nhev: [0]
                 v: [array([ 4.613e-09, -3.896e-09,  2.555e-09])]
            method: tr_interior_point
        optimality: 0.002435981079436409
  constr_violation: 0.0
    execution_time: 2439.46755194664
         tr_radius: 5.272835838258389e-09
    constr_penalty: 1.0
 barrier_parameter: 2.048000000000001e-09
 barrier_tolerance: 2.048000000000001e-09
             niter: 407


may have also worked for 4 total burn case? fixed t1, but could have let it be free
no cost on dispersion, objective is 30% worse than with free t1 from paper
           message: `gtol` termination condition is satisfied.
           success: True
            status: 1
               fun: 0.9077987770809444
                 x: [ 4.597e+03 -5.166e+03  1.209e+02  3.698e+02  7.183e+03
                     -5.498e+03 -9.271e+01  2.684e+02]
               nit: 195
              nfev: 184
              njev: 184
              nhev: 0
          cg_niter: 599
      cg_stop_cond: 4
              grad: [ 4.063e-09  5.255e-10  1.230e-09 -1.564e-09  2.034e-09
                      9.449e-10  1.213e-09 -2.280e-09]
   lagrangian_grad: [ 4.063e-09  5.255e-10  1.230e-09 -1.564e-09  2.034e-09
                      9.449e-10  1.213e-09 -2.280e-09]
            constr: [array([ 4.597e+03,  2.586e+03,  7.183e+03])]
               jac: [<3x8 sparse matrix of type '<class 'numpy.float64'>'
                    	with 4 stored elements in Compressed Sparse Row format>]
       constr_nfev: [0]
       constr_njev: [0]
       constr_nhev: [0]
                 v: [array([-4.466e-13, -7.947e-13,  4.251e-13])]
            method: tr_interior_point
        optimality: 4.0630590995652806e-09
  constr_violation: 0.0
    execution_time: 870.5520279407501
         tr_radius: 1005602.9416134846
    constr_penalty: 1.0
 barrier_parameter: 2.048000000000001e-09
 barrier_tolerance: 2.048000000000001e-09
             niter: 195

"""

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

class Deterministic3Burn1a(co.OptimizationProblem):
    rds = []
    n_burns = len(make_burn.burns)
    burn_config = dict(tig_1=first_tig)
    ts = [first_tig]
    for burn_num, burn, next_burn in zip(range(1,n_burns+2), make_burn.burns, make_burn.burns[1:]):
        ratio = burn_num/(n_burns+1)
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

opt2 = Deterministic3Burn1a()

#opt = Burn1(sigma_Dv_weight=3, mag_Dv_weight=1, sigma_r_weight=0)
#opt_sim = Sim(
#        tig_1=opt.t1,
#        **sim_kwargs
#)

print("\n"*3,"burn time minimization")
print(opt._stats)

