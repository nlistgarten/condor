import casadi
import condor as co

def flatten(symbol):
    if isinstance(symbol, co.FreeSymbol) and (symbol.diagonal or symbol.symmetric):
        raise NotImplemented
    return [
        elem
        for col in casadi.horzsplit(symbol.backend_repr)
        for elem in casadi.vertsplit(col)
    ]

def wrap(symbol, values):
    pass

class ExplicitSystem:
    def __init__(self, model):
        symbol_inputs = [
            elem for inp in model.input._symbols for elem in flatten(inp)
        ]
        symbol_outputs = [
            elem for out in model.output._symbols for elem in flatten(out)
        ]
        name_inputs = model.input.list_of('name')
        name_outputs = model.output.list_of('name')
        doc = f"{tuple(name_outputs)} = {model}{tuple(name_outputs)}"
        self.func =  casadi.Function(model.__name__, symbol_inputs, symbol_outputs)


    def __call__(self, *args, **kwargs):
        pass


class AlgebraicSystem:
    #  actually most of these should be classes? at least attach model

    #bound_behavior = enum(
    #    ignore,
    #    # I think backtracking behavior can be referred to as L2 for "vector" and L1 for
    #    # "scalar"  
    #)
    #
    #line_search_criteria = enum(
    ## wolf,?
    #    # or just true or false? 
    #)
    #
    def __init__(
        model, atol=1E-12, rtol=1E-12, warm_start=True, exact_hessian=True,
    ):
        return
        rootfinder_options = dict(
        )
        self.x = casadi.vertcat(*[
            flatten(out) for out in model.implicit_output.list_of('symbol')
        ])
        self.g0 = casadi.vertcat(*[
            flatten(out) for out in model.residual.list_of('symbol')
        ])
        self.g1 = casadi.vertcat(*[
            flatten(out) for out in model.explicit_output.list_of('symbol')
        ])
        self.p = casadi.vertcat(*[
            flatten(out) for out in model.input.list_of('symbol')
        ])

        self.model = model
        self.rootfinder = casadi.Rootfinder()

    def __call__(self, *args):
        p_arg = casadi.vertcat(args)


