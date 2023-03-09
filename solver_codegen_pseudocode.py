#
# in loop over om systems starting at prob.model
#
for omsys in model.system_iter:
    upsys = UpcycyleSystem()

    upsys.inputs = omsys.inputs
    internal_connections = upsys.solver.get_internal_connections(upsys.inputs)
    # solver only gets external connections to this system, could happen in get_internal_connections
    upsys.solver.inputs += ~internal_connections

    if omsys is explicit and omssys.is_forced_explicit:
        for output_name in omsys:
            upsys.outputs[output_name] = omsys.compute()  # residiual expr + output symbol

    else omsys is implicit:
        upsys.solver.residuals += omsys.residuals

#
# once a solver is built by the loop above
#

# Generated code:
def solver_system({external_inputs}):
    p = external_inputs
    resid_guesses_initial = warm_start()  # initial pass pulls defaults from om metadata

    def resid(external_inputs, resid_guesses):  # guesses are vals for solver.residuals.keys()
        # template generating explicit_output_var = rhs expr
        for varname, expr in solver.outputs.items():
            varname = expr

        for varname, expr in solver.residuals.items():
            varname + "_resid" = expr

        return varnames, varnames_resid

    resid_with_external_inputs = partial(resid, external_inputs=p)

    lbg = [-inf] * len(solver.outputs) + [0] * len(solver.residuals)
    ubg = [inf] * len(solver.outputs) + [0] * len(solver.residuals)

    out = solve(
        f=0,
        g=resid_with_external_inputs,
        x0=resid_guesses_initial,
        lbx=prob_meta[resid_guesses],
        ubx=prob_meta[resid_gueses],
        lbg=lbg,
        ubg=ubg
    )

    return out["g"][:len(solver.outputs) + out["x"]
