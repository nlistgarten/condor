import numpy as np
import condor as co
import casadi as ca


I6 = ca.MX.eye(6)
Z6 = ca.MX(6, 6)
W = ca.vertcat(I6, Z6)

#I3 = np.eye(3)
#Z3 = np.zeros((3,3))
#V = np.vstack((Z3, I3, np.zeros((6,3))))
#V = ca.sparsify(V)

I3 = ca.MX.eye(3)
Z3 = ca.MX(3,3)
V = ca.vertcat(Z3, I3, ca.MX(6,3))


numeric_constants = False
class LinCovCW(co.ODESystem):
    if numeric_constants:
        omega = 0.0011
        scal_w = ca.MX.ones(6)
        scal_v = ca.MX.ones(3)
    else:
        omega = parameter()
        scal_w = parameter(shape=6)
        scal_v = parameter(shape=3)

    initial_x = parameter(shape=6)
    initial_C = parameter(shape=(12,12))

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
    Acw[4,1] = -2*omega
    Acw[5,2] = 3*omega**2
    Acw[5,3] = -2*omega

    x = state(shape=6)
    C = state(shape=(12,12))
    Delta_v_mag = state()
    Delta_v_disp = state()

    Scal_w = ca.diag(scal_w)
    Cov_prop_offset = W @ Scal_w @ W.T

    Scal_v = ca.diag(scal_v)
    Cov_ctrl_offset = V @ Scal_v @ V.T

    Fcal = ca.MX(12,12)
    Fcal[:6, :6] = Acw
    Fcal[6:, 6:] = Acw

    initial[x] = initial_x
    initial[C] = initial_C

    dot[x] = Acw @ x
    dot[C] = Fcal @ C + C @ Fcal.T + Cov_prop_offset


sin = ca.sin
cos = ca.cos


class MajorBurn(LinCovCW.Event):
    rd = parameter(shape=3) # desired position
    tig = parameter() # time ignition
    tem = parameter() # time end maneuver

    function = t - tig
    at_time = [tig]

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

    update[Delta_v_mag] = Delta_v_mag + ca.norm_2(Delta_v)

    Mc = DG @ ca.horzcat(Z6, I6)
    sigma_Dv__2 = ca.trace( Mc @ C @ Mc.T)

    update[Delta_v_disp] = Delta_v_disp + ca.sqrt(sigma_Dv__2)


class Terminate(LinCovCW.Event):
    terminate = True
    # TODO: how to make a symbol like this just provide the backend repr? or is this
    # correct?
    function = t - MajorBurn.tem.backend_repr
    at_time = [MajorBurn.tem.backend_repr]


class Sim(LinCovCW.TrajectoryAnalysis):
    # TODO: add final burn Delta v (assume final relative v is 0, can get magnitude and
    # dispersion)
    tot_Delta_v_mag = trajectory_output(Delta_v_mag)
    tot_Delta_v_disp = trajectory_output(Delta_v_disp)

    #tf = parameter()

    Mr = ca.horzcat(I3, ca.MX(3,9))
    sigma_r__2 = ca.trace(Mr @ C @ Mr.T)
    final_pos_disp = trajectory_output(ca.sqrt(sigma_r__2))

    class Casadi(co.Options):
        use_lam_te_p = True
        include_lam_dot = False
        # old result possibly used lam(te_p) for tf but lam(te_m) for ti?

        #use_lam_te_p = False

        integrator_options = dict(
            #name="dop853",
            #rtol = 1E-9,
            #atol = 1E-12,
            nsteps = 10_000,
            max_step = 30.,
        )

from scipy.io import loadmat
Cov_0_matlab = loadmat('P_aug_0.mat')['P_aug_0'][0]

sim_kwargs = dict(
    omega = 0.00114,
    scal_w=[0.]*3 + [4.8E-10]*3,
    scal_v=[2.5E-7]*3,
    initial_x=[-2000., 0., 1000., 1.71, 0., 0.,],
    initial_C=Cov_0_matlab,
    rd=[500., 0., 0.],
)


from condor.backends.casadi.implementations import OptimizationProblem
class Hohmann(co.OptimizationProblem):
    tig = 84.
    tig = variable(initializer=200.)
    tf = variable(initializer=500.)
    constraint(tf - tig, lower_bound=30.)
    constraint(tig, lower_bound=0.)
    sim = Sim(
        tig=tig,
        tem=tf,
        **sim_kwargs
    )

    objective = sim.tot_Delta_v_mag #+ 3*sim.tot_Delta_v_disp

    class Casadi(co.Options):
        exact_hessian=False
        method = OptimizationProblem.Method.scipy_trust_constr

sol = Hohmann()

import sys
sys.exit()

sim = Sim(
    **sim_kwargs,
    tig=200.,
    tem=500.,
)
jac = sim.implementation.callback.jac_callback(sim.implementation.callback.p, [])

sim = Sim(
    **sim_kwargs,
    tig=84.08964425,
    tem=2839.86361908,
)
old_sol_jac = sim.implementation.callback.jac_callback(sim.implementation.callback.p, [])
old_sol_sim = sim


sim = Sim(
    **sim_kwargs,
    tig=sol.tig,
    tem=sol.tf,
)
new_sol_jac = sim.implementation.callback.jac_callback(sim.implementation.callback.p, [])
new_sol_sim = sim

