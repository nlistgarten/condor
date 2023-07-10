import ndsplines
import casadi
import numpy as np
from condor.backends.casadi.utils import CasadiFunctionCallbackMixin

class NDSplinesJacobianCallback(CasadiFunctionCallbackMixin, casadi.Callback):
    def has_jacobian(self):
        return False

    def __init__(self, antiderivative, interpolant, name, inames, onames, opts):
        casadi.Callback.__init__(self)
        self.antiderivative_callback = antiderivative
        self.i = antiderivative.i
        self.name = name
        self.inames = inames
        self.onames = onames
        self.opts = opts
        self.construct(name, {})
        self.interpolant = interpolant

    def get_n_in(self):
        return 2

    def get_n_out(self):
        return 1

    def get_sparsity_in(self,i):
        anti_deriv_interp = self.antiderivative_callback.interpolant
        if i==0: # nominal input
            return casadi.Sparsity.dense(anti_deriv_interp.xdim)
        elif i==1: # nominal output
            return casadi.Sparsity(anti_deriv_interp.ydim, 1)

    def get_sparsity_out(self,i):
        anti_deriv_interp = self.antiderivative_callback.interpolant
        return casadi.Sparsity.dense(anti_deriv_interp.ydim, anti_deriv_interp.xdim)

    def eval(self, args):
        return tuple(self.interpolant(args[0]))


class NDSplinesCallback(CasadiFunctionCallbackMixin, casadi.Callback):
    def __init__(self, intermediate):
        casadi.Callback.__init__(self)
        self.name = name = intermediate.model.__name__
        self.func = casadi.Function(
            f"{intermediate.model.__name__}_placeholder",
            intermediate.symbol_inputs,
            intermediate.symbol_outputs,
            dict(allow_free=True)
        )
        self.i = intermediate
        self.construct(name, {})

        model = self.i.model

        self.interpolant = ndsplines.make_interp_spline(
            self.i.input_data,
            self.i.output_data,
            degrees = self.i.degrees
        )

    def eval(self, args):
        return tuple(self.interpolant(args))

    def get_jacobian(self, name, inames, onames, opts):
        deriv_interpolants = [
            self.interpolant.derivative(idx) for idx in range(self.interpolant.xdim)
        ]
        jac_knots = [
            deriv_interpolants[idx].knots[idx] for idx in range(self.interpolant.xdim)
        ]
        jac_degrees = [
            deriv_interpolants[idx].degrees[idx] for idx in range(self.interpolant.xdim)
        ]
        jac_coefficients = np.stack([
            deriv_interpolant.coefficients
            for deriv_interpolant in deriv_interpolants
        ], axis=-1)
        interp = ndsplines.NDSpline(
            jac_knots, jac_coefficients, jac_degrees,
            self.interpolant.periodic, self.interpolant.extrapolate
        )
        self.jac_callback = NDSplinesJacobianCallback(self, interp, name,
                                                     inames, onames, opts)
        return self.jac_callback

"""
import condor as co
import numpy as np
class MyInterp(co.Tablelookup):
    x = input()
    y = output()

    input_data[x] = np.arange(10)*0.1
    output_data[y] = (input_data[x]-0.8)**2


class MyOpt(co.OptimizationProblem):
    xx = variable()
    objective = MyInterp(xx).y

    class Casadi(co.Options):
        exact_hessian = False

MyInterp.implementation.callback.jac_callback(0.3, 0.)

"""
