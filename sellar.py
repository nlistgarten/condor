import aesara
from aesara import tensor
from aesara.compile import io
from scipy import optimize

# x = tensor.scalar("x")
# z1 = tensor.scalar("z1")
# z2 = tensor.scalar("z2")
x = tensor.vector("x")
y2_in = tensor.scalar("y2")

y1 = x[1] ** 2 + x[2] + x[0] - 0.2 * y2_in
y2_out = tensor.sqrt(y1) + x[1] + x[2]

obj = x[0] ** 2 + x[1] + y1 + tensor.exp(-y2_out)

con1 = y1 - 3.16
con2 = 24.0 - y2_out
conc = y2_out - y2_in

# don't think this is quite right
ins = [x, io.In(y2_in, value=1.0, implicit=True)]

f_obj = aesara.function(ins, obj)
f_con1 = aesara.function(ins, con1)
f_con2 = aesara.function(ins, con2)
f_conc = aesara.function(ins, conc)

# problem:
# minimize obj wrt x1, z1, z2
# subject to con1 <= 0, con2 <= 0, 0 < x < 10, 0 < z1 < 10, 0 < z2 < 10

res = optimize.minimize(
    f_obj,
    [1.0, 5.0, 2.0],
    args=(),
    method="SLSQP",
    jac=None,
    bounds=[(0, 10), (0, 10), (0, 10)],
    constraints=[
        {"type": "ineq", "fun": f_con1},
        {"type": "ineq", "fun": f_con2},
        {"type": "eq", "fun": f_conc},
    ],
)
print(res)
