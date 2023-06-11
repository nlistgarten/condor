import numpy as np
import condor as co
import casadi as ca
import matplotlib.pyplot as plt

from condor.backends.casadi.implementations import OptimizationProblem
from LinCovCW import LinCovCW, make_burn, sim_kwargs, I3, Z3, I6, Z6

MajorBurn = make_burn(
    rd = LinCovCW.parameter(shape=3), # desired position
    tig = LinCovCW.parameter(), # time ignition
    tem = LinCovCW.parameter(), # time end maneuver
)


class Terminate(LinCovCW.Event):
    terminate = True
    # TODO: how to make a symbol like this just provide the backend repr? or is this
    # correct?
    function = t - MajorBurn.tem
    at_time = [MajorBurn.tem]

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
    function = t - meas_t_offset

# 1-burn sim
class Sim(LinCovCW.TrajectoryAnalysis):
    # TODO: add final burn Delta v (assume final relative v is 0, can get magnitude and
    # dispersion)
    tot_Delta_v_mag = trajectory_output(Delta_v_mag)
    tot_Delta_v_disp = trajectory_output(Delta_v_disp)


    Mr = ca.horzcat(I3, ca.MX(3,9))
    sigma_r__2 = ca.trace(Mr @ C @ Mr.T)
    final_pos_disp = trajectory_output(ca.sqrt(sigma_r__2))

sim_kwargs.update(dict(
    tem_1 = 2300.,
    meas_dt = 2300.,
    meas_t_offset = 851.,
))


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

opt = Burn1(sigma_Dv_weight=3, mag_Dv_weight=1, sigma_r_weight=0)
opt_sim = Sim(
        tig_1=opt.t1,
        **sim_kwargs
)

print("\n"*3,"burn time minimization")
print(opt._stats)

