import upcycle  # isort: skip

import os
import pprint
import re
import warnings

import numpy as np
import openmdao.api as om
import pandas as pd
from openmdao.api import IndepVarComp, Problem
from pycycle.api import AIR_JETA_TAB_SPEC
from pycycle.elements.ambient import Ambient
from pycycle.elements.combustor import Combustor
from pycycle.elements.flow_start import FlowStart
from pycycle.mp_cycle import Cycle

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


upsolver, prob = upcycle.upcycle_problem(make_problem)

inputs = np.hstack([upcycle.get_val(prob, absname) for absname in upsolver.inputs])
out = upsolver.run(*inputs)
print(out)

prob.run_model()

cols = ("name", "om_val", "ca_val")
vals = []
for name, ca_val in zip(upsolver.outputs, out):
    om_val = upcycle.get_val(prob, name)
    vals.append([name, om_val, ca_val])

df = pd.DataFrame(vals, columns=cols)


print(df[~np.isclose(df["om_val"], df["ca_val"], rtol=0, atol=5e-3)])
# without warm start, rtol 1E-4 passes everything but with warm start it doesn't??
print(df[~np.isclose(df["om_val"], df["ca_val"], rtol=1E-4, atol=0.)])


