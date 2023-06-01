import numpy as np
import condor as co
import casadi as ca
import matplotlib.pyplot as plt

from condor.backends.casadi.implementations import OptimizationProblem

I6 = ca.MX.eye(6)
Z6 = ca.MX(6, 6)
W = ca.vertcat(I6, Z6)

I3 = ca.MX.eye(3)
Z3 = ca.MX(3,3)
V = ca.vertcat(Z3, I3, ca.MX(6,3))

Wonb = ca.MX.eye(6)
Vonb = ca.vertcat(Z3, I3)

class LinCovCW(co.ODESystem):
    omega = parameter()
    scal_w = parameter(shape=6) # covariance elements for propagation
    scal_v = parameter(shape=3) # covariance elements for control update
    # estimated covariance elements for navigation covariance propagation and control
    scalhat_w = parameter(shape=6)
    scalhat_v = parameter(shape=3)

    initial_x = parameter(shape=6)
    initial_C = parameter(shape=(12,12))
    initial_P = parameter(shape=(6,6))

    Acw = ca.MX(6,6)

    """[

    [0, 0, 0,            1, 0, 0],
    [0, 0, 0,            0, 1, 0],
    [0, 0, 0,            0, 0, 1],

    [0, 0, 0,            0, 0, 2*omega],
    [0, -omega**2, 0,    0, 0, 0],
    [0, 0, 3*omega**2,   -2*omega, 0, 0]

    ] """

    Acw[0,3] = 1
    Acw[1,4] = 1
    Acw[2,5] = 1

    Acw[3,5] = 2*omega
    Acw[4,1] = -omega**2
    Acw[5,2] = 3*omega**2
    Acw[5,3] = -2*omega

    Delta_v_mag = state() # accumulator for Delta-v magnitude
    Delta_v_disp = state() # accumulator for Delta-v dispersion
    x = state(shape=6) # true state position and velocity
    C = state(shape=(12,12)) # augmented covariance
    P =  state(shape=(6,6)) # onboard covariance for navigation system (Kalman filter)



    Scal_w = ca.diag(scal_w)
    Cov_prop_offset = W @ Scal_w @ W.T

    Scal_v = ca.diag(scal_v)
    Cov_ctrl_offset = V @ Scal_v @ V.T

    Scalhat_w = ca.diag(scalhat_w)
    P_prop_offset = Wonb @ Scalhat_w @ Wonb.T

    Scalhat_v = ca.diag(scalhat_v)
    P_ctrl_offset = Vonb @ Scalhat_v @ Vonb.T

    Fcal = ca.MX(12,12)
    Fcal[:6, :6] = Acw
    Fcal[6:, 6:] = Acw

    initial[x] = initial_x
    initial[C] = initial_C
    initial[P] = initial_P

    dot[x] = Acw @ x
    dot[C] = Fcal @ C + C @ Fcal.T + Cov_prop_offset

    # TODO: in generla case, this should be a dfhat/dx(hat) instead of exact Acw
    # and should be reflected in bottom right corner of Fcal as well
    dot[P] = Acw @ P + P @ Acw.T + P_prop_offset


sin = ca.sin
cos = ca.cos

class Measurements(LinCovCW.Event):
    meas_dt = parameter()
    meas_t_offset = parameter()
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

    update[Delta_v_disp] = Delta_v_disp# + ca.sqrt(sigma_Dv__2)
    update[Delta_v_mag] = Delta_v_mag# + ca.sqrt(sigma_Dv__2)
    update[x] = x
    #function = ca.sin(np.pi*(t-meas_t_offset)/meas_dt)
    function = t - meas_t_offset

def make_burn(rd, tig, tem):
    burn_name = "Burn%d" % (1 + sum([
        event.__name__.startswith("Burn") for event in LinCovCW.Event.subclasses
    ]))
    attrs = co.InnerModelType.__prepare__(burn_name, (LinCovCW.Event,))
    update = attrs["update"]
    x = attrs["x"]
    C = attrs["C"]
    P = attrs["P"]
    t = attrs["t"]
    omega = attrs["omega"]
    Cov_ctrl_offset = attrs["Cov_ctrl_offset"]
    P_ctrl_offset = attrs["P_ctrl_offset"]
    Delta_v_mag = attrs["Delta_v_mag"]
    Delta_v_disp = attrs["Delta_v_disp"]

    if not LinCovCW.parameter.get(backend_repr=rd).name:
        attrs["rd_%d" % (1 + sum(
            [name.startswith("rd_") for name in LinCovCW.parameter.list_of('name')]
        ))] = rd

    if not LinCovCW.parameter.get(backend_repr=tig).name:
        attrs["tig_%d" % (1 + sum(
            [name.startswith("tig_") for name in LinCovCW.parameter.list_of('name')]
        ))] = tig

    if not LinCovCW.parameter.get(backend_repr=tem).name:
        attrs["tem_%d" % (1 + sum(
            [name.startswith("tem_") for name in LinCovCW.parameter.list_of('name')]
        ))] = tem

    attrs["function"] = t - tig
    attrs["at_time"] = [tig]

    t_d = tem - tig
    stm = ca.MX(6,6)
    stm[0,0] = 1
    stm[0,2] = 6*omega*t_d - 6*sin(omega*t_d)
    stm[0,3] = -3*t_d + 4*sin(omega*t_d)/omega
    stm[0,5] = 2*(1 - cos(omega*t_d))/omega
    stm[1,1] = cos(omega*t_d)
    stm[1,4] = sin(omega*t_d)/omega
    stm[2,2] = 4 - 3*cos(omega*t_d)
    stm[2,3] = 2*(cos(omega*t_d) - 1)/omega
    stm[2,5] = sin(omega*t_d)/omega
    stm[3,2] = 6*omega*(1 - cos(omega*t_d))
    stm[3,3] = 4*cos(omega*t_d) - 3
    stm[3,5] = 2*sin(omega*t_d)
    stm[4,1] = -omega*sin(omega*t_d)
    stm[4,4] = cos(omega*t_d)
    stm[5,2] = 3*omega*sin(omega*t_d)
    stm[5,3] = -2*sin(omega*t_d)
    stm[5,5] = cos(omega*t_d)
    T_pp = stm[:3, :3]
    T_pv = stm[:3, 3:]
    T_pv_inv = ca.solve(T_pv, ca.MX.eye(3))

    Delta_v = (T_pv_inv @ rd - T_pv_inv@T_pp @ x[:3, 0]) - x[3:, 0]
    update[Delta_v_mag] = Delta_v_mag + ca.norm_2(Delta_v)
    update[x]  = x + ca.vertcat(Z3, I3) @ (Delta_v)

    DG = ca.vertcat(
        ca.MX(3,6),
        ca.horzcat(-(T_pv_inv@T_pp), -I3)
    )
    Dcal = ca.vertcat(
        ca.horzcat(I6, DG),
        ca.horzcat(Z6, I6 + DG),
    )

    update[C] = Dcal @ C @ Dcal.T + Cov_ctrl_offset
    # TODO: in general case, this requires something like the bottom right corner of
    # Dcal which should use onboard models of control instead of exact control
    update[P] = P + P_ctrl_offset

    Mc = DG @ ca.horzcat(Z6, I6)
    sigma_Dv__2 = ca.trace( Mc @ C @ Mc.T)

    update[Delta_v_disp] = Delta_v_disp + ca.sqrt(sigma_Dv__2)
    Burn = co.InnerModelType(burn_name, (LinCovCW.Event,), attrs=attrs)

    Burn.rd = rd
    Burn.tig = tig
    Burn.tem = tem

    Burn.DG = DG
    Burn.Dcal = Dcal
    Burn.sigma_Dv__2 = sigma_Dv__2
    Burn.Mc = Mc
    return Burn

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

from scipy.io import loadmat
Cov_0_matlab = loadmat('P_aug_0.mat')['P_aug_0'][0]

sim_kwargs = dict(
    omega = 0.00114,
    # env.translation_process_noise_disp from IC_Traj_demo_000c.m
    scal_w=[0.]*3 + [4.8E-10]*3,
    #env.translation_maneuvers.var_noise_disp
    scal_v=[2.5E-7]*3,
    # env.translation_process_noise_err from IC_Traj_demo_000c.m
    scalhat_w=[0.]*3 + [4.8E-10]*3,
    #scalhat_w=[0.]*6,
    # env.translation_maneuvers.var_noise_err
    scalhat_v=[2.5E-7]*3,
    # io.R = env.sensors.rel_pos.measurement_var; from rel_pos.m which is set by
    # sig = 1e-3*(40/3); pos_var = [(sig)^2    (sig)^2    (sig)^2]; in rel_pos_ic.m
    rcal=[(1e-3*(40/3))**2]*3,
    # io.R_onb = io.onboard.sensors.rel_pos.measurement_var from rel_pos.m
    # in runRelMotionSetup.m, io.onboard = io.environment
    rcalhat=[(1e-3*(40/3))**2]*3,

    rd_1=[500., 0., 0.],

    meas_dt = 2300.,
    meas_t_offset = 851.,

    #meas_dt = 100.,
    #meas_t_offset = 51.,
)
sim_kwargs.update(dict(
    initial_x=[-2000., 0., 1000., sim_kwargs['omega']*1000.*3/2, 0., 0.,],
    initial_C=Cov_0_matlab,
    initial_P=Cov_0_matlab[-6:, -6:] - Cov_0_matlab[:6, :6],
))

# 1-burn sim
class Sim(LinCovCW.TrajectoryAnalysis):
    # TODO: add final burn Delta v (assume final relative v is 0, can get magnitude and
    # dispersion)
    tot_Delta_v_mag = trajectory_output(Delta_v_mag)
    tot_Delta_v_disp = trajectory_output(Delta_v_disp)

    #tf = parameter()

    Mr = ca.horzcat(I3, ca.MX(3,9))
    sigma_r__2 = ca.trace(Mr @ C @ Mr.T)
    final_pos_disp = trajectory_output(ca.sqrt(sigma_r__2))

    #class Casadi(co.Options):

    #    integrator_options = dict(
    #        rtol = 1E-9,
    #        atol = 1E-15,
    #        nsteps = 10_000,
    #        #max_step = 5.,
    #    )

sim_kwargs.update(dict(
    tem_1 = 2300.,
))


class Meas1(co.OptimizationProblem):
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

opt = Meas1(sigma_Dv_weight=3, mag_Dv_weight=1, sigma_r_weight=0)
opt_sim = Sim(
        tig_1=opt.t1,
        **sim_kwargs
)
"""
turning debug_level to 0 for shooting_gradient_method moves from 200s to 190s.

"""
import sys
sys.exit()


tigs = np.arange(10, 2200., 50)
sims= [
    Sim(
        tig_1=tig,
        **sim_kwargs
    )
    for tig in tigs
]
sims1 = sims

Dv_mags = [sim.tot_Delta_v_mag for sim in sims]
Dv_disps = [sim.tot_Delta_v_disp for sim in sims]
pos_disps = [sim.final_pos_disp for sim in sims]
ordinates = [Dv_mags, Dv_disps, pos_disps]

fig, axes = plt.subplots(3, constrained_layout=True, sharex=True)
for ax, ordinate in zip(axes, ordinates):
    ax.plot(tigs, ordinate)
    ax.grid(True)

# 2-burn sim
sim_kwargs.update(dict(
    tem_1 = 1210.,
    tig_1 = 10.,
))
MinorBurn = make_burn(
    rd = MajorBurn.rd, # desired position
    tig = LinCovCW.parameter(), # time ignition
    tem = MajorBurn.tem, # time end maneuver
)

class Sim2(LinCovCW.TrajectoryAnalysis):
    # TODO: add final burn Delta v (assume final relative v is 0, can get magnitude and
    # dispersion)
    tot_Delta_v_mag = trajectory_output(Delta_v_mag)
    tot_Delta_v_disp = trajectory_output(Delta_v_disp)

    #tf = parameter()

    Mr = ca.horzcat(I3, ca.MX(3,9))
    sigma_r__2 = ca.trace(Mr @ C @ Mr.T)
    final_pos_disp = trajectory_output(ca.sqrt(sigma_r__2))

    #class Casadi(co.Options):

    #    integrator_options = dict(
    #        rtol = 1E-9,
    #        atol = 1E-15,
    #        nsteps = 10_000,
    #        #max_step = 5.,
    #    )

tigs = np.arange(600, 1100., 25)
sims= [
    Sim2(
        tig_2=tig,
        **sim_kwargs
    )
    for tig in tigs
]
sims2 = sims

Dv_mags = [sim.tot_Delta_v_mag for sim in sims]
Dv_disps = [sim.tot_Delta_v_disp for sim in sims]
pos_disps = [sim.final_pos_disp for sim in sims]
ordinates = [Dv_mags, Dv_disps, pos_disps]

fig, axes = plt.subplots(3, constrained_layout=True, sharex=True)
for ax, ordinate in zip(axes, ordinates):
    ax.plot(tigs, ordinate)
    ax.grid(True)


plt.show()




#sim1 = Sim(
#        tem_1=opt.tf,
#        tig_1=850,
#        **sim_kwargs
#)
#
#
#sim2 = Sim(
#        tem_1=opt.tf,
#        tig_1=852,
#        **sim_kwargs
#)

