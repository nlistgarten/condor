import aesara.tensor as at
from aesara import function as fn
from scipy import optimize

xx = at.vector("xx")
x, z1, z2, y2_in = [xx[i] for i in range(4)]

@sysin(x, z1, z2, y2)
@sysout(y1)
def y1__y2(x, z1, z2, y2):
    return z1**2 + z2 + x - 0.2 * y2

class sys1(...):
    def compute?(x, z1, z2, y2):
        return ...

    y1 = Outputs
    y2,y3 = compute

def sys0(x, z1, z2, y2):
    return z1**2 + z2 + x - 0.2 * y2

copy1 = system(sys0, "y1")
copy2 = system(sys0, "y1")

@system("y1")
def sys1(x, z1, z2, y2):
    return z1**2 + z2 + x - 0.2 * y2

@system("y2")
def sys2(y1, z1, z2):
    return at.sqrt(y1) + z1 + z2

@system("obj")
def objective(x, y1, y2, z2)
    return x**2 + z2 + y1 + at.exp(-y2)

@system("obj", "constr1", "constr2")
def sellar(x, z1, z2):
    y1 = sys1(x=x, z1=z1, z2=z2, y2=sys2.y2)
    y2 = sys2(x=x, z1=z1, z2=z2, y1=sys1.y1)

    obj = x**2 + z2 + y1 + at.exp(-y2)
    constr1 = y1 > 3.16
    constr2 = y2 < 24.
    return obj, constr1, constr2

prob.add_objective(sellar.obj)
prob.add_constraints(sellar.constr1:)

objective = cp.Minimize(sellar.obj)
prob = cp.Problem(obj, sellar.constr1:)

def block_diagram(t, x):
    xdot, y = plant(t, x, u=controller.u)
    u = controller(t, y=plant.y)
    return xdot, (y, u)


class Sellar(...):
    subsystems = [sys1, sys2] # do we need this?
    x = fw.designvar #basically declaring inputs
    z1 = ...

    fw.connect(x, [sys1.x, sys2.x])# can programmatically do this
    fw.connect(z1, [sys1.z1, sys2.z1])




    obj = x + sys1.y2

    def objective(self):
        x**2 + z2 + y1 + at.exp(-y2_in)
        self.x


obj = x**2 + z2 + y1 + at.exp(-y2_in)
con1 = y1 - 3.16
con2 = 24.0 - y2_out
con_y2 = y2_out - y2_in

ins = [xx]

res = optimize.minimize(
    fn(ins, obj),
    [1.0, 5.0, 2.0, 1.0],
    method="SLSQP",
    #jac=fn(ins, at.grad(obj, xx)),
    jac=None,
    bounds=[(0, 10), (0, 10), (0, 10), (None, None)],
    constraints=[
        {"type": "ineq", "fun": fn(ins, con1),}, # "jac": fn(ins, at.grad(con1, ins))},
        {"type": "ineq", "fun": fn(ins, con2),}, # "jac": fn(ins, at.grad(con2, ins))},
        {"type": "eq", "fun": fn(ins, con_y2),}, # "jac": fn(ins, at.grad(con_y2, ins))},
    ],
    tol=1e-8,
)
print(res)

print("checking...")
print("y1 =", y1.eval({xx: res.x}))
print("y2 =", y2_out.eval({xx: res.x}))
