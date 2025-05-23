#from numpy import *
import numpy as np
import casadi
import condor.backends.casadi as backend
# useful but not sure if all backends would have:
# symvar -- list all symbols present in expression
# depends_on
#

pi = casadi.pi
inf = casadi.inf
nan = np.nan

mod = casadi.fmod

atan = casadi.atan
atan2 = casadi.atan2
tan = casadi.tan
sin = casadi.sin
cos = casadi.cos
asin = casadi.asin
acos = casadi.acos
exp = casadi.exp
log = casadi.log
log10 = casadi.log10
sqrt = casadi.sqrt

eye = casadi.MX.eye
ones = casadi.MX.ones

def diag(v, k=0):
    if k != 0:
        raise ValueError("Not supported for this backend")
    if not hasattr(v, "shape"):
        # try to concat list/tuple of elements
        v = concat(v)
    return casadi.diag(v)

def vector_norm(x, ord=2):
    if ord==2:
        return casadi.norm_2(x)
    if ord==1: return casadi.norm_1(x)
    if ord==inf:
        return casadi.norm_inf(x)

solve = casadi.solve

def concat(arrs, axis=0):
    """ implement concat from array API for casadi """
    if not arrs:
        return arrs
    if np.any([isinstance(arr, backend.symbol_class) for arr in arrs]):
        if axis == 0:
            return casadi.vcat(arrs)
        elif axis in (1,-1):
            return casadi.hcat(arrs)
        else:
            raise ValueError("casadi only supports matrices")
    else:
        return np.concat([np.atleast_2d(arr) for arr in arrs], axis=axis)

def unstack(arr, axis=0):
    if axis == 0:
        return casadi.vertsplit(arr)
    elif axis in (1, -1):
        return casadi.horzsplit(arr)

def zeros(shape=(1,1)):
    return backend.symbol_class(*shape)



def min(x, axis=None):
    if not isinstance(x, backend.symbol_class):
        x = concat(x)
    if axis is not None:
        raise ValueError("Only axis=None supported")
    return casadi.mmin(x)

def max(x, axis=None):
    if not isinstance(x, backend.symbol_class):
        x = concat(x)
    if axis is not None:
        raise ValueError("Only axis=None supported")
    return casadi.mmax(x)


def jacobian(of, wrt):
    """ jacobian of expression `of` with respect to symbols `wrt` """
    """
    we can apply jacobian to ExternalSolverWrapper but it's a bit clunky because need
    symbol_class expressions for IO, and to evalaute need to create a Function. Not sure
    how to create a backend-generic interface for this. When do we want an expression vs
    a callable? Maybe the overall process is right (e.g., within an optimization
    problem, will have a variable flat input, and might just want the jac_expr)

    Example to extend from docs/howto_src/table_basicsa.py

       flat_inp = SinTable.input.flatten()
       wrap_inp = SinTable.input.wrap(flat_inp)
       instance = SinTable(**wrap_inp.asdict()) # needed so callback object isn't destroyed
       wrap_out = instance.output
       flat_out = wrap_out.flatten()
       jac_expr = ops.jacobian(flat_out, flat_inp)
       from condor import backend
       jac = backend.expression_to_operator(flat_inp, jac_expr, "my_jac")
       #jac = casadi.Function("my_jac", [flat_inp], [jac_expr])
       jac(0.)
    """
    if of.size:
        return casadi.jacobian(of, wrt)
    else:
        return casadi.MX()

def jac_prod(of, wrt, rev=True):
    """ create directional derivative """
    return casadi.jtimes(of, wrt, not rev)

def substitute(expr, subs):
    original_expr = expr
    for key, val in subs.items():
        try:
            expr = casadi.substitute(expr, key, val)
        except Exception as e:
            print(e)
            breakpoint()
            raise e
    return expr

    if isinstance(expr, backend.symbol_class):
        expr = casadi.substitute([expr], list(subs.keys()), list(subs.values()))[0]
    return expr

def recurse_if_else(*conditions_actions):
    """

    recurse_if_else(
        (condition0, action0),
        (codnition1, action1),
        ...
        else_action
    )


    conditions_actions is a list of
        (condition, action) pairs, which should get constructed in order

        to be equivalent to

        if condition0:
            action0
        elif condition1:
            action1
        ...

        the last element in the list may be a single else action

        else:
            else_action

    """
    if len(conditions_actions) == 1:
        if not hasattr(conditions_actions[0], "__len__"):
            return conditions_actions[0]
        else:
            return conditions_actions[0][0]
    condition, action = conditions_actions[0]
    remainder = recurse_if_else(*conditions_actions[1:])
    return casadi.if_else(condition, action, remainder)



