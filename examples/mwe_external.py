import condor as co
import casadi
import os
import numpy as np


try:
    import spiceypy as spice
except:
    pass
else:

    os.chdir("/Users/bmargoli/projects/condor-flight/checkcases/lunar/")
    spice.furnsh("mrotat.tm")

    def left_quaternion_product_matrix(q):
        return np.array([
            [-q[1], -q[2], -q[3]],
            [q[0], -q[3], q[2]],
            [q[3], q[0], -q[1]],
            [-q[2], q[1], q[0]],
        ]).squeeze()


    #class SpiceReferenceFrame(type):
    class SpiceReferenceFrame(co.ExternalSolverWrapper, parameterized_IO=False):
        #dt = input()
        #q = output(shape=4)
        #omega = output(shape=3)

        #def __new__(cls, model_name, bases, attrs,  **kwargs):
        #    new_cls = super().__new__(cls, model_name, bases, attrs, **kwargs)
        #    return new_cls


        #def __call__(cls, inertial_frame_name, name, start_time_string):
        def __init__(self, inertial_frame_name, name, start_time_string):
            """
            or...
            cls.dt = cls.input(name="dt")
            cls.q = cls.input(shape=4, name="q")
            cls.omega = cls.input(shape=3, name="omega")

            OR
            allow something along the lines of "auto_create" flag, or figure it out based on
            context? lol

            cls.input(name="dt")
            cls.input(shape=4, name="q")
            cls.input(shape=3, name="omega")
            """
            self.input(name="dt")
            self.output(name="q", shape=4)
            self.output(name="omega", shape=3)

            self.inertial_frame_name = inertial_frame_name
            self.name = name
            self.start_time_string = start_time_string
            self.et_start = spice.str2et(start_time_string)
            #self.create_model()

        def function(cls, dt):
            # need cls ref to access things like inertial_frame_name, etc
            et = cls.et_start + dt
            name = cls.name
            SS = spice.sxform(cls.inertial_frame_name, cls.name, et)
            RR, omega_other = spice.xf2rav(SS)
            ang_vel = RR @ omega_other
            #quaternion = spice.m2q(RR.T.copy())
            #quaternion = spice.m2q(RR.copy())
            quaternion = spice.m2q(RR)
            quaternion[1:] *= -1
            return quaternion, ang_vel


        # allow jac_vec_prod, hessian, etc
        # okay, I guess a finite_difference_mixin would be easy to add -- 
        def jacobian(cls, t):
            q, ang_vel = cls.function(t)
            qjac = left_quaternion_product_matrix(q) @ ang_vel/2
            return qjac, np.zeros(3)

    spice_reference_frame = SpiceReferenceFrame(
        "J2000", "moon_pa", "2026-1-28 6:42:03.51 UTC"
    )
    at_0 = spice_reference_frame(0)
    print(at_0.input, at_0.output)

    class Solve(co.AlgebraicSystem):
        dt = implicit_output(initializer=-50_000)
        sys = spice_reference_frame(dt)
        residual.qq = casadi.norm_2(at_0.q - casadi.vertcat(*sys.q.squeeze().tolist()))
        #residual.qq = casadi.DM(at_0.q) - sys.q

    solve = Solve()
    print(
        solve.dt
    )




data_yy = dict(
    sigma = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.1833, 0.1621, 0.1429, 0.1256, 0.1101, 0.0966],
            [0.4, 0.3600, 0.3186, 0.2801, 0.2454, 0.2147, 0.1879],
            [0.6, 0.5319, 0.4654, 0.4053, 0.3526, 0.3070, 0.2681],
            [0.8, 0.6896, 0.5900, 0.5063, 0.4368, 0.3791, 0.3309],
            [1.0, 0.7857, 0.6575, 0.5613, 0.4850, 0.4228, 0.3712],
        ]
    ),
    sigstr = np.array(
        [
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.04, 0.7971, 0.9314, 0.9722, 0.9874, 0.9939, 0.9969],
            [0.16, 0.7040, 0.8681, 0.9373, 0.9688, 0.9839, 0.9914],
            [0.36, 0.7476, 0.8767, 0.9363, 0.9659, 0.9812, 0.9893],
            [0.64, 0.8709, 0.9338, 0.9625, 0.9778, 0.9865, 0.9917],
            [1.0, 0.9852, 0.9852, 0.9880, 0.9910, 0.9935, 0.9954],
        ]
    ),
)
data_xx = dict(
    xbbar = np.linspace(0, 1.0, data_yy["sigma"].shape[0]),
    xhbar = np.linspace(0, 0.3, data_yy["sigma"].shape[1]),
)
Table = co.TableLookup(data_xx, data_yy, 1)
tt = Table(0.5, 0.5)
print(tt.input, tt.output)

class MyOpt3(co.OptimizationProblem):
    xx = variable(warm_start=False)
    yy = variable(warm_start=False)
    interp = Table(xx, yy)
    objective = (interp.sigma - 0.2)**2 + (interp.sigstr - .7)**2

    class Options:
        exact_hessian = False
        exact_hessian = True
        print_level = 0

opt3 = MyOpt3()
print(opt3.implementation.stats['iter_count'])

MyOpt3.Options.exact_hessian = False

opt3 = MyOpt3()
print(opt3.implementation.stats['iter_count'])

MyOpt3.Options.exact_hessian = True

opt3 = MyOpt3()
print(opt3.implementation.stats['iter_count'])
