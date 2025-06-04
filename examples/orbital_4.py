import sys

import casadi as ca
import numpy as np
from LinCovCW import I3, I6, Z3, Z6, LinCovCW, make_burn, make_sim

import condor as co
from condor.backends.casadi.implementations import OptimizationProblem


class Terminate(LinCovCW.Event):
    terminate = True
    terminate_time = parameter()
    # TODO: how to make a symbol like this just provide the backend repr? or is this
    # correct?
    # function = t - terminate_time
    at_time = (terminate_time,)


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
    # function = meas_dt*ca.sin(np.pi*(t-meas_t_offset)/meas_dt)/np.pi
    # function = t - meas_t_offset
    at_time = slice(meas_t_offset, None, meas_dt)


def add_measurement_params(kwargs):
    kwargs.update(
        dict(
            scalhat_w=kwargs["scal_w"],
            scalhat_v=kwargs["scal_v"],
            rcalhat=kwargs["rcal"],
            initial_P=kwargs["initial_C"][-6:, -6:] - kwargs["initial_C"][:6, :6],
        )
    )


def make_C0(pos_disp, vel_disp, nav_pos_err=None, nav_vel_err=None):
    if nav_pos_err is None:
        nav_pos_err = pos_disp
    if nav_vel_err is None:
        nav_vel_err = vel_disp
    D0 = np.diag(np.r_[pos_disp, vel_disp])
    P0 = np.diag(np.r_[nav_pos_err, nav_vel_err])
    return np.block([[D0, D0], [D0, D0 + P0]])


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
alt_km = 400.0
r_orbit_m = (r_earth_km + alt_km) * 1e3

base_kwargs = dict(
    omega=np.sqrt(mu_earth / ((r_orbit_m) ** 3)), meas_dt=10.0, meas_t_offset=1.0
)

scenario_1_target = [-200.0, 0.0, 0.0]
scenario_1_kwargs = dict(
    #  (0 km; −10 km; 0.2 km; 0 m∕s; 0 m∕s; 0 m∕s)
    # The initial relative states (radial, along-track, and cross-track) are
    # x0 (0 km; −10 km; 0.2 km; 0 m∕s; 0 m∕s; 0 m∕s), and the desired final relative
    # states are xf (0 km;−0.2 km;0 km;0 m∕s;0 m∕s;0 m∕s)
    # CW frame in meters:
    # x is down-track with -x behind, +x in front
    # y is cross-track,
    # z altitude with +z below, -z above
    initial_x=[
        -10_000.0,
        200,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
)

scenario_2_target = [200.0, 0.0, 0.0]
scenario_2_kwargs = dict(
    # The initial relative states (radial, along-track, and cross- track) are x0
    # (1 km; 10 km; 0.2 km; 0 m∕s; −1.7 m∕s; 0.2 m∕s, and the desired final relative
    # states are xf (0 km; 0.2 km; 0 km; 0 m∕s; 0 m∕s; 0 m∕s).
    initial_x=[
        10_000.0,
        200.0,
        1_000.0,
        1.71,
        0.0,
        0.0,
    ],
)

low_cost_kwargs = dict(
    scal_w=[0.0] * 6,
    initial_C=make_C0([1000.0] * 3, [1] * 3),
    # measurement error
    rcal=[10.0] * 3,
    # control variance
    scal_v=[0.1] * 3,
)
add_measurement_params(low_cost_kwargs)


nominal_kwargs = dict(
    scal_w=[0.0] * 6,
    initial_C=make_C0([100.0] * 3, [0.1] * 3),
    # measurement error
    rcal=[1.0] * 3,
    # control variance
    scal_v=[0.01] * 3,
)
add_measurement_params(nominal_kwargs)


high_cost_kwargs = dict(
    scal_w=[0.0] * 6,
    initial_C=make_C0([10.0] * 3, [0.01] * 3),
    # measurement error
    rcal=[0.1] * 3,
    # control variance
    scal_v=[0.001] * 3,
)
add_measurement_params(high_cost_kwargs)


MajorBurn = make_burn(
    rd=LinCovCW.parameter(shape=3),  # desired position
    tig=LinCovCW.parameter(),  # time ignition
    tem=LinCovCW.parameter(),  # time end maneuver
)

# 1-burn sim
Sim = make_sim()

scenario_a_tf = 2000.0
scenario_b_tf = 12_000.0
first_tig = 0.0  # TODO: handle this....
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

# base_kwargs['meas_t_offset'] = scenario_b_tf*2

# generate additional burns
for idx in range(1):
    MajorBurn = make_burn(
        rd=LinCovCW.parameter(shape=3),  # desired position
        tig=LinCovCW.parameter(),  # time ignition
        tem=LinCovCW.parameter(),  # time end maneuver
    )

Sim = make_sim()


class OptimizeBurns(co.OptimizationProblem):
    disp_weighting = parameter()
    rds = []
    n_burns = len(make_burn.burns)
    t_1 = variable(initializer=1.0)
    ts = [t_1]
    burn_config = dict(tig_1=t_1)
    constraint(t_1, lower_bound=0.0)
    for burn_num, burn, next_burn in zip(
        range(1, n_burns + 2), make_burn.burns, make_burn.burns[1:]
    ):
        ratio = burn_num / n_burns
        rds.append(
            variable(
                name=f"target_pos_{burn_num}",
                shape=(3,),
                initializer=np.array(scenario_kwargs["initial_x"][:3]) * (1 - ratio)
                + np.array(scenario_target) * ratio,
            )
        )
        ts.append(
            variable(
                name=f"t_{burn_num + 1}",
                initializer=scenario_kwargs["terminate_time"] * ratio,
            )
        )
        constraint(ts[-1] - ts[-2], lower_bound=10.0)
        burn_config[LinCovCW.parameter.get(backend_repr=burn.rd).name] = rds[-1]
        burn_config[LinCovCW.parameter.get(backend_repr=burn.tem).name] = ts[-1]
        burn_config[LinCovCW.parameter.get(backend_repr=next_burn.tig).name] = ts[-1]

    # constraint(ts[0], lower_bound=0.)
    # for tig1, tig2 in zip(ts, ts[1:]):
    #    constraint(tig2 - tig1, lower_bound=10.)
    # del tig1
    # del tig2

    constraint(ts[-1], upper_bound=scenario_kwargs["terminate_time"])
    burn_config[LinCovCW.parameter.get(backend_repr=next_burn.rd).name] = (
        scenario_target
    )
    burn_config[LinCovCW.parameter.get(backend_repr=next_burn.tem).name] = (
        scenario_kwargs["terminate_time"]
    )
    print(burn_config)

    sim = Sim(**base_kwargs, **scenario_kwargs, **cost_system_kwargs, **burn_config)
    objective = sim.final_vel_mag + disp_weighting * sim.final_vel_disp
    for burn_num in range(n_burns):
        objective += getattr(
            sim, f"Delta_v_mag_{burn_num + 1}"
        ) + disp_weighting * getattr(sim, f"Delta_v_disp_{burn_num + 1}")

    class Casadi(co.Options):
        exact_hessian = False
        method = OptimizationProblem.Method.scipy_trust_constr
        scipy_trust_constr_options = dict(xtol=0.5)


##########
# print basic results
#########

print([sim.final_pos_disp for sim in scenario_1a])
print([sim.final_vel_disp + sim.Delta_v_disp_1 for sim in scenario_1a])
print([sim.final_vel_mag + sim.Delta_v_mag_1 for sim in scenario_1a])

print([sim.final_pos_disp for sim in scenario_2a])
print([sim.final_vel_disp + sim.Delta_v_disp_1 for sim in scenario_2a])
print([sim.final_vel_mag + sim.Delta_v_mag_1 for sim in scenario_2a])


sys.exit()
opt = OptimizeBurns(disp_weighting=0.0)

burn_config = {
    k: (
        getattr(opt, OptimizeBurns.variable.get(backend_repr=v).name)
        if isinstance(v, co.backend.symbol_class)
        else v
    )
    for k, v in OptimizeBurns.burn_config.items()
}
# base_kwargs['meas_t_offset'] = 1.
sim = Sim(**base_kwargs, **scenario_kwargs, **cost_system_kwargs, **burn_config)
n_burns = len(make_burn.burns)
print(f"number of burns: {n_burns}")
sum_Dv_mag = 0.0
sum_Dv_disp = 0.0
for burn_num in range(1, n_burns + 1):
    Dv_mag = getattr(sim, f"Delta_v_mag_{burn_num}")
    Dv_disp = getattr(sim, f"Delta_v_disp_{burn_num}")
    tig = getattr(sim, f"tig_{burn_num}")
    print(f"burn {burn_num} at time {tig} Dv mag={Dv_mag}  Dv_disp={Dv_disp}")
    sum_Dv_mag += Dv_mag
    sum_Dv_disp += Dv_disp
print(f"station keeping Dv mag: {sim.final_vel_mag} Dv disp: {sim.final_vel_disp}")
sum_Dv_mag += sim.final_vel_mag
sum_Dv_disp += sim.final_vel_disp
print(f"sum Dv mag: {sum_Dv_mag} sum Dv disp: {sum_Dv_disp}")


"""
it seems like deterministic optimization works? optimized without measurements and no
cost on dispersions. printing results:
all with nominal system

number of burns: 4
burn 1 at time 0.0 Dv mag=0.3153309307059405  Dv_disp=0.7758637078444677
burn 2 at time 3177.8613106217754 Dv mag=0.14203863154665575  Dv_disp=3.111739412027804
burn 3 at time 6006.860129115569 Dv mag=0.00010573804293171246  Dv_disp=3.082904533529533
burn 4 at time 8936.682407426808 Dv mag=0.13910781100635025  Dv_disp=0.6227367795704676
station keeping Dv mag: 0.34438088985547344 Dv disp: 0.8457393966256312
sum Dv mag: 0.9409640011573517 sum Dv disp: 8.438983829597904
               nit: 83
              nfev: 79
              njev: 79
              nhev: 0
          cg_niter: 206
      cg_stop_cond: 4
    execution_time: 557.3584508895874
                 x: [ 3.178e+03 -7.822e+03 -1.173e+02  8.452e+02  6.007e+03
                     -5.116e+03  1.046e+02  2.467e+02  8.937e+03 -2.342e+03
                     -6.541e+01  8.439e+02]
number of burns: 3
burn 1 at time 0.0 Dv mag=0.3190782349198173  Dv_disp=0.7753267283700211
burn 2 at time 4597.286167918477 Dv mag=0.09632063881042148  Dv_disp=2.09186927785427
burn 3 at time 7183.471779707452 Dv mag=0.12307501820671937  Dv_disp=2.4635927604678933
station keeping Dv mag: 0.3693249362442293 Dv disp: 0.6033972893243764
sum Dv mag: 0.9077988281811875 sum Dv disp: 5.934186056016561
               nit: 157
              nfev: 147
              njev: 147
              nhev: 0
          cg_niter: 465
    execution_time: 783.8889710903168
                 x: [ 4.597e+03 -5.166e+03  1.209e+02  3.699e+02  7.183e+03
                     -5.498e+03 -9.270e+01  2.684e+02]

number of burns: 2
burn 1 at time 0.0 Dv mag=0.4769377976734905  Dv_disp=0.7763192505627957
burn 2 at time 6005.014128295938 Dv mag=4.030104489162445e-05  Dv_disp=1.1096163206930227
station keeping Dv mag: 0.5278931124949962 Dv disp: 0.7114305651416224
sum Dv mag: 1.0048712112133782 sum Dv disp: 2.597366136397441
               nit: 55
              nfev: 51
              njev: 51
              nhev: 0
          cg_niter: 61
    execution_time: 172.77766108512878
                 x: [ 6.005e+03 -5.102e+03  1.131e+02 -7.700e+01]


free t1 found a much better answer that matches the paper SOCP sort of
number of burns: 3
burn 1 at time 1209.2894545866598 Dv mag=0.33766356512284995  Dv_disp=0.8092825243719685
burn 2 at time 4874.401911309868 Dv mag=0.030531473412992758  Dv_disp=0.9324664218268841
burn 3 at time 7780.871126412133 Dv mag=0.04809948465444022  Dv_disp=0.8153536524167068
station keeping Dv mag: 0.2962967042513561 Dv disp: 0.5804081468585136
sum Dv mag: 0.712591227441639 sum Dv disp: 3.137510745474073
               nit: 111
              nfev: 101
              njev: 101
              nhev: 0
          cg_niter: 402
    execution_time: 402.08212018013
                 x: [ 1.209e+03 -5.691e+03  3.252e+01  7.442e+02  4.874e+03
                     -4.935e+03 -3.871e+01  4.180e+02  7.781e+03]

more burns isn't better with a coast
number of burns: 4
burn 1 at time 15.058356538734472 Dv mag=0.3207376172108785  Dv_disp=0.5994442544823134
burn 2 at time 3216.4773005350257 Dv mag=0.13587308145536162  Dv_disp=1.6161122192921535
burn 3 at time 6044.639240368033 Dv mag=3.813559192315825e-05  Dv_disp=1.6485758290639116
burn 4 at time 8923.919031666028 Dv mag=0.13875794652737028  Dv_disp=0.6249323665270865
station keeping Dv mag: 0.34797000880727824 Dv disp: 0.8391191198047925
sum Dv mag: 0.9433767895928119 sum Dv disp: 5.328183789170257
               nit: 106
              nfev: 102
              njev: 102
              nhev: 0
          cg_niter: 265
    execution_time: 864.384425163269

with stochastic aware:
umber of burns: 3
burn 1 at time 31.082632071856537 Dv mag=0.21903784717380712  Dv_disp=0.5527441482614776
burn 2 at time 4151.451077246527 Dv mag=0.5520327023769721  Dv_disp=0.18215686105777934
burn 3 at time 7872.1051405466915 Dv mag=0.5732367858859878  Dv_disp=0.4712048659592569
station keeping Dv mag: 0.23714228875344903 Dv disp: 0.5865132804600679
sum Dv mag: 1.581449624190216 sum Dv disp: 1.7926191557385818
               nit: 46
              nfev: 43
              njev: 43
              nhev: 0
          cg_niter: 77
      cg_stop_cond: 2
    execution_time: 21758.865051031113
                 x: [ 3.108e+01 -6.730e+03 -7.218e+01  3.660e+02  4.151e+03
                     -3.466e+03 -1.066e+02  3.646e+02  7.872e+03]
was running 3 opts simultaneously, I think I killed my memory. Seemed to finish pretty
quick after that.

"""  # noqa

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

"""  # noqa

scenario_kwargs = dict(
    **scenario_1_kwargs,
    terminate_time=scenario_b_tf,
)


MajorBurn = make_burn(
    rd=LinCovCW.parameter(shape=3),  # desired position
    tig=LinCovCW.parameter(),  # time ignition
    tem=LinCovCW.parameter(),  # time end maneuver
)
Sim = make_sim()


class Deterministic3Burn1a(co.OptimizationProblem):
    rds = []
    n_burns = len(make_burn.burns)
    burn_config = dict(tig_1=first_tig)
    ts = [first_tig]
    for burn_num, burn, next_burn in zip(
        range(1, n_burns + 2), make_burn.burns, make_burn.burns[1:]
    ):
        ratio = burn_num / (n_burns + 1)
        ts.append(
            variable(
                name=f"t_{burn_num + 1}",
                initializer=scenario_kwargs["terminate_time"] * ratio,
            )
        )
        rds.append(
            variable(
                name=f"pos_{burn_num + 1}",
                shape=(3,),
                initializer=np.array(scenario_kwargs["initial_x"][:3]) * (1 - ratio)
                + np.array(scenario_target) * ratio,
            )
        )
        constraint(ts[-1] - ts[-2], lower_bound=10.0)
        burn_config[LinCovCW.parameter.get(backend_repr=burn.tem).name] = ts[-1]
        burn_config[LinCovCW.parameter.get(backend_repr=next_burn.tig).name] = ts[-1]
        burn_config[LinCovCW.parameter.get(backend_repr=burn.rd).name] = rds[-1]
    constraint(ts[-1], upper_bound=scenario_kwargs["terminate_time"])
    burn_config[LinCovCW.parameter.get(backend_repr=next_burn.tem).name] = (
        scenario_kwargs["terminate_time"]
    )
    burn_config[LinCovCW.parameter.get(backend_repr=next_burn.rd).name] = (
        scenario_target
    )

    sim = Sim(**base_kwargs, **scenario_kwargs, **cost_system_kwargs, **burn_config)
    objective = sim.final_vel_mag + 3 * sim.final_vel_disp
    for burn_num in range(n_burns):
        objective += getattr(sim, f"Delta_v_mag_{burn_num + 1}") + 3 * getattr(
            sim, f"Delta_v_disp_{burn_num + 1}"
        )

    class Casadi(co.Options):
        exact_hessian = False
        # method = OptimizationProblem.Method.scipy_trust_constr


opt2 = Deterministic3Burn1a()

# opt = Burn1(sigma_Dv_weight=3, mag_Dv_weight=1, sigma_r_weight=0)
# opt_sim = Sim(
#        tig_1=opt.t1,
#        **sim_kwargs
# )

print("\n" * 3, "burn time minimization")
print(opt._stats)
