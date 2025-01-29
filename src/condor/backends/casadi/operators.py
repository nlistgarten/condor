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

def jacobian(of, wrt):
    """ create a callable that computes dense jacobian """
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
