import upcycle  # isort: skip

import os
import pprint
import re
import warnings

import casadi
import numpy as np
import pandas as pd
import sympy as sym
from openmdao.api import IndepVarComp, Problem
from pycycle.elements.combustor import Combustor
from pycycle.elements.flow_start import FlowStart
from pycycle.elements.ambient import Ambient
from pycycle.mp_cycle import Cycle
from pycycle.api import AIR_JETA_TAB_SPEC
# from pycycle.thermo.cea import species_data

header = [
    "Fl_I.W",
    "Fl_I.Pt",
    "Fl_I.Tt",
    "Fl_I.ht",
    "Fl_I.s",
    "Fl_I.MN",
    "FAR",
    "eff",
    "Fl_O.MN",
    "Fl_O.Pt",
    "Fl_O.Tt",
    "Fl_O.ht",
    "Fl_O.s",
    "Wfuel",
    "Fl_O.Ps",
    "Fl_O.Ts",
    "Fl_O.hs",
    "Fl_O.rhos",
    "Fl_O.gams",
]

h_map = dict(((v_name, i) for i, v_name in enumerate(header)))

data = np.array(
    [
        38.8,
        158.428,
        1278.64,
        181.381769,
        1.690022166,
        0.3,
        0.02673,
        1,
        0.2,
        158.428,
        2973.240078,
        176.6596564,
        1.959495309,
        1.037124,
        154.436543,
        2956.729659,
        171.467604,
        0.1408453453,
        1.279447105,
    ]
)

os.environ["OPENMDAO_REPORTS"] = "none"


def make_problem():
    prob = Problem()
    model = prob.model = Cycle()
    model.options["thermo_method"] = "TABULAR"
    # model.options["thermo_data"] = species_data.janaf
    # FUEL_TYPE = "Jet-A(g)"
    model.options['thermo_data'] = AIR_JETA_TAB_SPEC
    FUEL_TYPE = "FAR"

    model.add_subsystem(
        "ivc",
        IndepVarComp(
            "in_composition",
            [3.23319235e-04, 1.10132233e-05, 5.39157698e-02, 1.44860137e-02],
        ),
    )

    model.add_subsystem("flow_start", FlowStart())
    model.add_subsystem("combustor", Combustor(fuel_type=FUEL_TYPE))
    # model.add_subsystem("ambient", Ambient())

    model.pyc_connect_flow("flow_start.Fl_O", "combustor.Fl_I")

    # model.set_input_defaults('Fl_I:tot:P', 100.0, units='lbf/inch**2')
    # model.set_input_defaults('Fl_I:tot:h', 100.0, units='Btu/lbm')
    # model.set_input_defaults('Fl_I:stat:W', 100.0, units='lbm/s')
    model.set_input_defaults("combustor.Fl_I:FAR", 0.0)
    model.set_input_defaults("combustor.MN", 0.5)


    # needed because composition is sized by connection
    # model.connect('ivc.in_composition', ['Fl_I:tot:composition', 'Fl_I:stat:composition', ])

    prob.set_solver_print(level=-1)
    prob.setup(check=False, force_alloc_complex=True)


    # input flowstation
    # prob['Fl_I:tot:P'] = data[h_map['Fl_I.Pt']]
    # prob['Fl_I:tot:h'] = data[h_map['Fl_I.ht']]
    # prob['Fl_I:stat:W'] = data[h_map['Fl_I.W']]
    # prob['Fl_I:FAR'] = data[h_map['FAR']]
    prob.set_val("flow_start.P", data[h_map["Fl_I.Pt"]], units="psi")
    prob.set_val("flow_start.T", data[h_map["Fl_I.Tt"]], units="degR")
    prob.set_val("flow_start.W", data[h_map["Fl_I.W"]], units="lbm/s")
    prob["combustor.Fl_I:FAR"] = data[h_map["FAR"]]
    prob["combustor.MN"] = data[h_map["Fl_O.MN"]]

    return prob

prob = make_problem()
up_prob = make_problem()
prob_after = make_problem()
prob.final_setup()
x0, lbx, ubx, iobj, non_indep_idxs, explicit_idxs, const_idxs = upcycle.extract_problem_data(prob)

prob.model._apply_nonlinear()
om_res = prob.model._residuals.asarray().copy()

prob.run_model()

x0_after = upcycle.extract_problem_data(prob)[0]

om_res_final = prob.model._residuals.asarray().copy()
prob.model._apply_nonlinear()
om_res_final2 = prob.model._residuals.asarray().copy()


# prob.model.combustor.mix_fuel.list_inputs(print_arrays=True)
# prob.model.combustor.mix_fuel.list_outputs(print_arrays=True)
# print(prob['combustor.Fl_I:tot:composition'])
# print(prob['combustor.Fl_I:tot:n'])
print(prob["combustor.Fl_I:tot:h"])
print(prob["combustor.Fl_I:tot:P"])
# exit()

# check outputs
tol = 1.0e-2

npss = data[h_map["Fl_O.Pt"]]
pyc = prob["combustor.Fl_O:tot:P"]
rel_err = abs(npss - pyc) / npss
print("Pt out:", npss, pyc, rel_err)

npss = data[h_map["Fl_O.Tt"]]
pyc = prob["combustor.Fl_O:tot:T"]
rel_err = abs(npss - pyc) / npss
print("Tt out:", npss, pyc, rel_err)

npss = data[h_map["Fl_O.ht"]]
pyc = prob["combustor.Fl_O:tot:h"]
rel_err = abs(npss - pyc) / npss
print("ht out:", npss, pyc, rel_err)

npss = data[h_map["Fl_O.Ps"]]
pyc = prob["combustor.Fl_O:stat:P"]
rel_err = abs(npss - pyc) / npss
print("Ps out:", npss, pyc, rel_err)

npss = data[h_map["Fl_O.Ts"]]
pyc = prob["combustor.Fl_O:stat:T"]
rel_err = abs(npss - pyc) / npss
print("Ts out:", npss, pyc, rel_err)

npss = data[h_map["Wfuel"]]
pyc = prob["combustor.Fl_I:stat:W"] * (prob["combustor.Fl_I:FAR"])
rel_err = abs(npss - pyc) / npss
print("Wfuel:", npss, pyc, rel_err)

print("")

# partial_data = prob.check_partials(out_stream=None, method='cs',
#                                    includes=['combustor.*',], excludes=['*.base_thermo.*', '*.mix_fuel.*'])
# assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

# warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)

print("upcycling...")
sym_prob, res_mat, out_syms = upcycle.upcycle_problem(up_prob)

print("sympy2casdadi")
ca_vars = casadi.vertcat(*[casadi.MX.sym(s.name) for s in out_syms])
ca_res, ca_vars_out = upcycle.sympy2casadi(res_mat, out_syms, ca_vars)

print("updating initial guesses for explicit outputs")
# update explicit output guesses
res = casadi.Function("res", casadi.vertsplit(ca_vars), casadi.vertsplit(ca_res))

df_res = pd.DataFrame(
    {
        "om_res": om_res[non_indep_idxs],
        "casadi": casadi.vertcat(*res(*x0)).toarray().squeeze(),
        "om_res_final": om_res_final[non_indep_idxs],
    },
    index=out_syms[non_indep_idxs],
)

xz = np.zeros_like(x0)
xz[non_indep_idxs] = np.array(res(*x0)).squeeze()
x0[explicit_idxs] += xz[explicit_idxs]

const_constr = casadi.vertcat(*[ca_vars[i] - x0[i] for i in const_idxs])

f = 0 if iobj is None else ca_vars[iobj]
nlp = {"x": ca_vars, "f": f, "g": casadi.vertcat(ca_res, const_constr)}
S = casadi.nlpsol("S", "ipopt", nlp)

print("running...")

out = S(x0=x0, lbx=lbx, ubx=ubx, lbg=0, ubg=0)

df_res["casadi_final"] = casadi.vertcat(*res(*out["x"].toarray())).toarray().squeeze()
df_res["casadi_final2"] = casadi.vertcat(*res(*x0_after)).toarray().squeeze()

df = pd.DataFrame(
    {
        "initial_value": x0,
        "upper_bound": ubx,
        "lower_bound": lbx,
        "solution_value": out["x"].toarray().squeeze(),
        "x0_after": x0_after,
    },
    index=out_syms,
)

pattern = re.compile(r"(.*)\[(\d)+\]$")

om_vals = []
for idx, row in df.iterrows():
    name = idx.name
    try:
        val = prob.get_val(name)[0]
        prob_after.set_val(name, row.x0_after)
    except KeyError:
        match = pattern.match(name)
        base_name = match[1]
        i = int(match[2])
        val = prob.get_val(base_name)[i]
        prob_after.set_val(base_name, row.x0_after, indices=i)

    om_vals.append(val)
df["om_vals"] = om_vals

pd.set_option("display.max_rows", None)
print(df)

prob_after.final_setup()
prob_after.model._apply_nonlinear()
om_res_after_solve = prob_after.model._residuals.asarray().copy()
df_res["om_res_after"] = om_res_after_solve[non_indep_idxs]
