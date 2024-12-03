import ndsplines
import casadi
import numpy as np
from condor.backends.casadi.utils import CasadiFunctionCallbackMixin


class NDSplinesHessianCallback(CasadiFunctionCallbackMixin, casadi.Callback):
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

        anti_deriv_interp = self.antiderivative_callback.interpolant[0]
        self.original_ydim = anti_deriv_interp.ydim
        self.original_xdim = anti_deriv_interp.xdim
        self.primary_shape = (
            anti_deriv_interp.ydim * anti_deriv_interp.xdim,
            anti_deriv_interp.xdim,
        )
        self.func = antiderivative.func.jacobian()
        self.construct(name, {})
        self.interpolant = interpolant

    # def get_sparsity_in(self,i):
    #     if i==0: # nominal input
    #         return casadi.Sparsity.dense(self.primary_shape[1])
    #     elif i==1: # nominal output
    #         return casadi.Sparsity(self.primary_shape[0], 1)
    #     elif i == 2:
    #         return casadi.Sparsity(self.antiderivative_callback.primary_shape)

    # def get_sparsity_out(self,i):
    #     return casadi.Sparsity.dense(self.primary_shape)

    def eval(self, local_args):
        breakpoint()
        array_vals = [
            interp(np.array(local_args[0]).reshape(-1))[0, :]
            for interp in self.interpolant
        ]
        return_val = np.stack(array_vals, axis=1)
        return casadi.DM(return_val), self.get_sparsity_out(1)


class NDSplinesJacobianCallback(CasadiFunctionCallbackMixin, casadi.Callback):
    def has_jacobian(self):
        return True

    def __init__(self, antiderivative, interpolant, name, inames, onames, opts):
        casadi.Callback.__init__(self)
        self.antiderivative_callback = antiderivative
        self.i = antiderivative.i
        self.name = name
        self.inames = inames
        self.onames = onames
        self.opts = opts

        anti_deriv_interp = self.antiderivative_callback.interpolant
        self.primary_shape = (anti_deriv_interp.ydim, anti_deriv_interp.xdim)

        self.func = antiderivative.func.jacobian()

        self.construct(name, {})
        self.interpolant = interpolant

    def get_sparsity_in(self, i):
        if i == 0:  # nominal input
            return casadi.Sparsity.dense(self.primary_shape[1])
        elif i == 1:  # nominal output
            return casadi.Sparsity(self.primary_shape[0], 1)

    def get_sparsity_out(self, i):
        return casadi.Sparsity.dense(self.primary_shape)

    def eval(self, local_args):
        array_vals = [
            interp(np.array(local_args[0]).reshape(-1))[0, :]
            for interp in self.interpolant
        ]
        return_val = np.stack(array_vals, axis=1)
        return (casadi.DM(return_val),)

    def get_jacobian(self, name, inames, onames, opts):
        interp = [
            interpolant.derivative(idx)
            if interpolant.degrees[idx] > 0
            else lambda *args: np.zeros(interpolant.ydim)
            for interpolant in self.interpolant
            for idx in range(interpolant.xdim)
        ]
        self.jac_callback = NDSplinesHessianCallback(
            self, interp, name, inames, onames, opts
        )
        return self.jac_callback


class NDSplinesCallback(CasadiFunctionCallbackMixin, casadi.Callback):
    def __init__(self, intermediate):
        casadi.Callback.__init__(self)
        self.name = name = intermediate.model.__name__
        self.func = casadi.Function(
            f"{intermediate.model.__name__}_placeholder",
            [casadi.vertcat(*intermediate.symbol_inputs)],
            [casadi.vertcat(*intermediate.symbol_outputs)],
            dict(allow_free=True),
        )
        self.i = intermediate
        self.construct(name, {})

        model = self.i.model

        self.interpolant = ndsplines.make_interp_spline(
            self.i.input_data, self.i.output_data, degrees=self.i.degrees
        )

    def eval(self, args):
        return (self.interpolant(np.array(args[0]).reshape(-1))[0, :],)

    def get_jacobian(self, name, inames, onames, opts):
        interp = [
            self.interpolant.derivative(idx) for idx in range(self.interpolant.xdim)
        ]
        self.jac_callback = NDSplinesJacobianCallback(
            self, interp, name, inames, onames, opts
        )
        return self.jac_callback
