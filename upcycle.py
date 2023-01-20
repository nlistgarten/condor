import numpy as np
import sympy as sym
from openmdao.vectors.default_vector import DefaultVector


class SymbolicArray(np.ndarray):
    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        print(ufunc.__name__)
        if method != "__call__":
            return NotImplemented

        sym_ufunc = getattr(sym, ufunc.__name__, None)
        if sym_ufunc is not None:
            print("using sympy ufunc")
            result = np.array([sym_ufunc(inp) for inp in inputs[0]]).view(SymbolicArray)

        else:
            # convert inputs that are type SymbolicArray to ndarray
            args = []
            for inp in inputs:
                args.append(
                    inp if not isinstance(inp, SymbolicArray) else inp.view(np.ndarray)
                )

            outputs = out
            out_no = []
            if outputs:
                out_args = []
                for j, output in enumerate(outputs):
                    if isinstance(output, SymbolicArray):
                        out_no.append(j)
                        out_args.append(output.view(np.ndarray))
                    else:
                        out_args.append(output)
                kwargs["out"] = tuple(out_args)
            else:
                outputs = (None,) * ufunc.nout

            result = super().__array_ufunc__(ufunc, method, *args, **kwargs)
            if result is NotImplemented:
                return NotImplemented

        if result is NotImplemented:
            return NotImplemented

        return result.view(SymbolicArray)  # result[0] if len(result) == 1 else result


class SymbolicVector(DefaultVector):
    def __getitem__(self, name):
        return np.atleast_1d(super().__getitem__(name))

    def _create_data(self):
        system = self._system()
        size = np.sum(system._var_sizes[self._typ][system.comm.rank, :])
        dtype = complex if self._alloc_complex else float
        # return np.zeros(size, dtype=dtype)

        names = []
        # om uses this and relies on ordering when building views, should be ok
        for abs_name, meta in system._var_abs2meta[self._typ].items():
            sz = meta["size"]
            if sz == 1:
                names.append(abs_name)
            else:
                names.extend([f"{abs_name}_{i}" for i in range(sz)])
        return np.array(sym.symbols(names)).view(SymbolicArray)

    def set_var(self, name, val, idxs=None, flat=False, var_name=None):
        pass
