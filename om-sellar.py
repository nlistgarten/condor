import upcycle
import numpy as np
import pandas as pd
import openmdao.api as om
import upcycle

deriv_method = "exact"  # or "fd"


class SellarDis1(om.ExplicitComponent):
    def setup(self):
        self.add_input("x", val=0.0)
        self.add_input("z", val=np.zeros(2))
        self.add_input("y2", val=1.0)
        self.add_output("y1", val=1.0, lower=0.1, upper=1000.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method=deriv_method)

    def compute(self, inp, out):
        out["y1"] = inp["z"][0] ** 2 + inp["z"][1] + inp["x"] - 0.2 * inp["y2"]

    def compute_partials(self, inp, J):
        J["y1", "y2"] = -0.2
        J["y1", "z"] = np.array([[2.0 * inp["z"][0], 1.0]])
        J["y1", "x"] = 1.0


class SellarDis2(om.ExplicitComponent):
    def setup(self):
        self.add_input("z", val=np.zeros(2))
        self.add_input("y1", val=1.0)
        self.add_output("y2", val=1.0, lower=0.1, upper=1000.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method=deriv_method)

    def compute(self, inp, out):
        out["y2"] = inp["y1"] ** 0.5 + inp["z"][0] + inp["z"][1]

    def compute_partials(self, inp, J):
        y1 = inp["y1"]
        J["y2", "y1"] = 0.5 * y1**-0.5
        J["y2", "z"] = np.array([[1.0, 1.0]])


class Sellar(om.Group):
    def setup(self):
        self.add_subsystem("d1", SellarDis1(), promotes=["x", "z", "y1", "y2"])
        self.add_subsystem("d2", SellarDis2(), promotes=["z", "y1", "y2"])

        self.add_subsystem(
            "obj_cmp",
            om.ExecComp(
                "obj = x**2 + z[1] + y1 + exp(-y2)",
                obj=0.0,
                x=0.0,
                z=np.array([0.0, 0.0]),
                y1=0.0,
                y2=0.0,
            ),
            promotes=["obj", "x", "z", "y1", "y2"],
        )

        self.add_subsystem(
            "con_cmp1",
            om.ExecComp("con1 = 3.16 - y1", con1=0.0, y1=0.0),
            promotes=["con1", "y1"],
        )
        self.add_subsystem(
            "con_cmp2",
            om.ExecComp("con2 = y2 - 24.0", con2=0.0, y2=0.0),
            promotes=["con2", "y2"],
        )

        self.set_input_defaults("x", 1.0)
        self.set_input_defaults("z", np.array([5.0, 2.0]))

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)


def make_problem():
    prob = om.Problem()
    prob.model = Sellar()

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"] = 1e-8

    prob.model.add_design_var("x", lower=0, upper=10)
    prob.model.add_design_var("z", lower=0, upper=10)
    prob.model.add_objective("obj")
    prob.model.add_constraint("con1", upper=0)
    prob.model.add_constraint("con2", upper=0)

    prob.setup()

    prob.set_val("x", 1.0)
    prob.set_val("z", [5.0, 2.0])
    prob.set_val("y1", 1.0)
    prob.set_val("y2", 1.0)

    return prob

updriver, prob = upcycle.upcycle_problem(make_problem)

### Check run_model
upsolver = updriver.children[0]
inputs = np.hstack([upcycle.get_val(prob, absname) for absname in upsolver.inputs])
out = upsolver(inputs)[0]

prob.set_solver_print(-1)
prob.run_model()

cols = ("name", "om_val", "ca_val")
vals = []
for name, ca_val in zip(upsolver.outputs, out):
    om_val = upcycle.get_val(prob, name)
    vals.append([name, om_val, ca_val])

df = pd.DataFrame(vals, columns=cols)

# passes with or without warm start
print(df[~np.isclose(df["om_val"], df["ca_val"], rtol=0., atol=1e-9)])
print(df[~np.isclose(df["om_val"], df["ca_val"], rtol=1e-10, atol=0.)])


### check optimizer

inputs = np.hstack([upcycle.get_val(prob, absname) for absname in updriver.inputs])
out = updriver(inputs)
print(out)


prob.set_solver_print(level=0)

#prob.run_driver()

#print("minimum found at")
#print(prob.get_val("x")[0])
#print(prob.get_val("z"))
#print(prob.get_val("y1")[0])
#print(prob.get_val("y2")[0])
#
#print("minumum objective")
#print(prob.get_val("obj")[0])
