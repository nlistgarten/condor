from condor.backends.casadi.utils import CasadiFunctionCallbackMixin
import casadi
import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve
from condor.solvers.shooting_gradient_method import ResultInterpolant


DEBUG_LEVEL = 0

class ShootingGradientMethodJacobian(CasadiFunctionCallbackMixin, casadi.Callback):
    def has_jacobian(self):
        return False

    def __init__(self, shot, name, inames, onames, opts):
        casadi.Callback.__init__(self)
        self.shot = shot
        self.i = shot.i
        self.name = name
        self.inames = inames
        self.onames = onames
        self.opts = opts
        self.construct(name, {})

    def get_n_in(self):
        return 2

    def get_n_out(self):
        return 1

    def get_sparsity_in(self,i):
        shot = self.shot
        p = shot.i.p
        out = shot.i.traj_out_expr
        if i==0: # nominal input
            return casadi.Sparsity.dense(p.shape)
        elif i==1: # nominal output
            return casadi.Sparsity(out.shape)

    def get_sparsity_out(self,i):
        p = self.i.p
        out = self.i.traj_out_expr
        return casadi.Sparsity.dense(out.shape[0], p.shape[0])

    def eval(self, args):
        p = args[0]
        if not casadi.is_equal(self.shot.p, args[0]):
            self.shot(args[0])
        o = args[1]
        jac = self.i.shooting_gradient_method(self.shot.res)

        #Sim = self.i.model
        #DV_idx = Sim.trajectory_output.flat_index(Sim.tot_Delta_v_mag)
        #tig_idx = Sim.parameter.flat_index(Sim.tig)
        #tem_idx = Sim.parameter.flat_index(Sim.tem)
        #p = p.toarray().squeeze()
        #print("jac params:", p[tig_idx], p[tem_idx])
        #print("jac:", jac[DV_idx, tig_idx], jac[DV_idx, tem_idx])

        return jac,

class ShootingGradientMethod(CasadiFunctionCallbackMixin, casadi.Callback):
    def __init__(self, intermediate):
        casadi.Callback.__init__(self)
        self.name = name = intermediate.model.__name__
        self.func = casadi.Function(
            f"{intermediate.model.__name__}_placeholder",
            [intermediate.p],
            [intermediate.traj_out_expr],
            dict(allow_free=True,),
        )
        self.i = intermediate
        self.construct(name, {})
        self.from_implementation = False

    def get_jacobian(self, name, inames, onames, opts):
        if DEBUG_LEVEL:
            print("\n"*10, f"getting jacobian for {self} . {name}", sep="\n")
            if hasattr(self, "p"):
                print(f"p={self.p}")
        self.jac_callback = ShootingGradientMethodJacobian(self, name, inames, onames, opts)
        return self.jac_callback

    def eval(self, args):
        p = self.p = casadi.vertcat(*args)
        self.res = self.i.StateSystem(p)
        self.output = self.i.trajectory_analysis(self.res)

        return self.output,

