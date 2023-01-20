"""Sellar problem using an ImplicitComponent instead of a cycle

Prototype for injecting sympy symbols in to get derivatives
"""
import numpy as np
import openmdao.api as om


class Resid(om.ImplicitComponent):
    def setup(self):
        self.add_input("x", val=0.0)
        self.add_input("z", val=np.zeros(2))

        self.add_output("y1")
        self.add_output("y2")

    def setup_partials(self):
        self.declare_partials("*", "*")

    def apply_nonlinear(self, inputs, outputs, residuals):
        x, z = inputs.values()
        y1, y2 = outputs.values()
        residuals["y1"] = y1 - z[0] ** 2 - z[1] - x + 0.2 * y2
        residuals["y2"] = y2 - y1**0.5 - z[0] - z[1]

    def linearize(self, inputs, outputs, partials):
        x, z = inputs.values()
        y1, y2 = outputs.values()

        partials["y1", "x"] = -1
        partials["y1", "z"] = [-2 * z[0], -1]
        partials["y1", "y1"] = 1
        partials["y1", "y2"] = 0.2

        partials["y2", "x"] = 0
        partials["y2", "z"] = [-1, -1]
        partials["y2", "y1"] = -0.5 * y1**-0.5
        partials["y2", "y2"] = 1


class Obj(om.ExplicitComponent):
    def setup(self):
        self.add_input("x", val=0.0)
        self.add_input("z", val=np.zeros(2))
        self.add_input("y1", val=0.0)
        self.add_input("y2", val=0.0)

        self.add_output("obj")

    def setup_partials(self):
        self.declare_partials("obj", ["*"])

    def compute(self, inputs, outputs):
        x, z, y1, y2 = inputs.values()
        outputs["obj"] = x**2 + z[1] + y1 + np.exp(-y2)

    def compute_partials(self, inputs, J):
        x, z, y1, y2 = inputs.values()
        J["obj", "x"] = 2 * x
        J["obj", "z"] = [0, 1]
        J["obj", "y1"] = 1
        J["obj", "y2"] = -np.exp(-y2)


class Constr(om.ExplicitComponent):
    def setup(self):
        self.add_input("y1", val=0.0)
        self.add_input("y2", val=0.0)

        self.add_output("con1")
        self.add_output("con2")

    def setup_partials(self):
        self.declare_partials("con1", ["y1"])
        self.declare_partials("con2", ["y2"])

    def compute(self, inputs, outputs):
        y1, y2 = inputs.values()
        outputs["con1"] = 3.16 - y1
        outputs["con2"] = y2 - 24.0

    def compute_partials(self, inputs, J):
        y1, y2 = inputs.values()
        J["con1", "y1"] = -1.0
        J["con2", "y2"] = 1.0


class Sellar(om.Group):
    def setup(self):
        self.add_subsystem("R", Resid(), promotes=["*"])
        self.add_subsystem("f", Obj(), promotes=["*"])
        self.add_subsystem("g", Constr(), promotes=["*"])

        self.set_input_defaults("x", 1.0)
        self.set_input_defaults("z", np.array([5.0, 2.0]))
        self.set_input_defaults("y2", 0.0)

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        # self.nonlinear_solver = om.NonlinearRunOnce()
        self.linear_solver = om.DirectSolver()


def make_sellar_problem():
    prob = om.Problem()
    prob.model = Sellar()

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"] = 1e-8
    # prob.driver.options["debug_print"] = ["objs"]

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
