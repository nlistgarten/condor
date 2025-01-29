from numpy import *
import casadi

pi = casadi.pi

def concat(arrs, axis=0):
    """ implement concat from array API for casadi """
    if axis == 0:
        return casadi.vcat(arrs)
    elif axis == 1:
        return casadi.hcat(arrs)
    else:
        raise ValueError("casadi only supports matrices")

def jacobian(of, wrt=None):
    """ create a callable that computes dense jacobian """
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
    return casadi.jacobian(of, wrt)

def jac_prod(of, wrt, rev=True):
    """ create directional derivative """
    return casadi.jtimes(of, wrt, not rev)

def substitute(expr, subs):
    for key, val in subs.items():
        expr = casadi.substitute(expr, key, val)
    return expr

def recurse_if_else(conditions_actions):
    if len(conditions_actions) == 1:
        return conditions_actions[0][0]
    condition, action = conditions_actions[-1]
    remainder = recurse_if_else(conditions_actions[:-1])
    return casadi.if_else(condition, action, remainder)
