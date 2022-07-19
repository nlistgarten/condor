import aesara.tensor as at
from aesara import function as fn
from scipy import optimize

xx = at.vector("xx")
x, z1, z2, y2_in = [xx[i] for i in range(4)]

y1 = z1**2 + z2 + x - 0.2 * y2_in
y2_out = at.sqrt(y1) + z1 + z2
obj = x**2 + z2 + y1 + at.exp(-y2_in)

con1 = y1 - 3.16
con2 = 24.0 - y2_out
con_y2 = y2_out - y2_in

ins = [xx]

res = optimize.minimize(
    fn(ins, obj),
    [1.0, 5.0, 2.0, 1.0],
    method="SLSQP",
    jac=fn(ins, at.grad(obj, xx)),
    bounds=[(0, 10), (0, 10), (0, 10), (None, None)],
    constraints=[
        {"type": "ineq", "fun": fn(ins, con1), "jac": fn(ins, at.grad(con1, ins))},
        {"type": "ineq", "fun": fn(ins, con2), "jac": fn(ins, at.grad(con2, ins))},
        {"type": "eq", "fun": fn(ins, con_y2), "jac": fn(ins, at.grad(con_y2, ins))},
    ],
    tol=1e-8,
)
print(res)

print("checking...")
print("y1 =", y1.eval({xx: res.x}))
print("y2 =", y2_out.eval({xx: res.x}))
