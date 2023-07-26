import numpy as np
import condor as co
import casadi as ca

from condor.backends.casadi.implementations import OptimizationProblem

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
    Acw[4,1] = -omega**2
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

def make_burn(rd, tig, tem):
    burn_name = "Burn%d" % (1 + sum([
        event.__name__.startswith("Burn") for event in LinCovCW.Event.subclasses
    ]))
    attrs = co.InnerModelType.__prepare__(burn_name, (LinCovCW.Event,))
    update = attrs["update"]
    x = attrs["x"]
    C = attrs["C"]
    t = attrs["t"]
    omega = attrs["omega"]
    Cov_ctrl_offset = attrs["Cov_ctrl_offset"]
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

    #attrs["function"] = t - tig
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


MinorBurn = make_burn(
    rd = MajorBurn.rd, # desired position
    tig = LinCovCW.parameter(), # time ignition
    tem = MajorBurn.tem, # time end maneuver
)

class Terminate(LinCovCW.Event):
    terminate = True
    # TODO: how to make a symbol like this just provide the backend repr? or is this
    # correct?
    #function = t - MajorBurn.tem
    at_time = [MajorBurn.tem]


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


from scipy.io import loadmat
Cov_0_matlab = loadmat('P_aug_0.mat')['P_aug_0'][0]

sim_kwargs = dict(
    omega = 0.00114,
    scal_w=[0.]*3 + [4.8E-10]*3,
    scal_v=[2.5E-7]*3,
    initial_x=[-2000., 0., 1000., 1.71, 0., 0.,],
    initial_C=Cov_0_matlab,
    rd_1=[500., 0., 0.],
)


class Geller2006(co.OptimizationProblem):
    t1 = 10.
    tf = 1210.
    t2 = variable(initializer=700.)
    sigma_r_weight = parameter()
    sigma_Dv_weight = parameter()


    constraint(t2, lower_bound=t1+600., upper_bound=tf-120.)

    sim = Sim(
        tig_1=t1,
        tig_2=t2,
        tem_1=tf,
        **sim_kwargs
    )
    objective = sigma_Dv_weight*sim.tot_Delta_v_disp + sigma_r_weight*sim.final_pos_disp

    class Casadi(co.Options):
        exact_hessian=False
        method = OptimizationProblem.Method.scipy_trust_constr


##############

r = Geller2006(sigma_r_weight = 1., sigma_Dv_weight = 0.)
v = Geller2006(sigma_r_weight = 0., sigma_Dv_weight = 500.)



sim = Sim(
    tig_1=Geller2006.t1,
    tig_2=700.,
    tem_1=Geller2006.tf,
    **sim_kwargs
)
#jac = sim.implementation.callback.jac_callback(sim.implementation.callback.p, [])
print("\n"*3, "minimize position dispersions:")
print(r._stats)

print("\n"*3, "minimize Delta-v dispersions:")
print(v._stats)


