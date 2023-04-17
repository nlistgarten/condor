import casadi

class Function:
    def __init__(self, model):
        symbol_inputs = [
            flatten(inp) for inp in  model.input.list_of('symbol')
        ]
        symbol_outputs = [
            flatten(out) for out in model.output.list_of('symbol')
        ]
        name_inputs = [
            flatten(inp) for inp in  model.input.list_of('name')
        ]
        name_outputs = [
            flatten(out) for out in model.output.list_of('name')
        ]
        doc = f"{tuple(name_outputs)} = {model}{tuple(name_outputs)}"
        self.func =  casadi.Function(symbol_inputs, symbol_outputs)


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
        model, atol, rtol, warm_start, exact_hessian
    ):
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


