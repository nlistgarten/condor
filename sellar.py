import aesara
import aesara.tensor as at
from aesara.compile import io
from scipy import optimize

xx = at.vector("xx")
x = xx[0]
z1 = xx[1]
z2 = xx[2]
y1_in = xx[3]
y2_in = xx[4]


y1_out = z1 ** 2 + z2 + x - 0.2 * y2_in
y2_out = at.sqrt(y1_in) + z1 + z2
obj = x ** 2 + z2 + y1_in + at.exp(-y2_in)

con1 = y1_in - 3.16
con2 = 24.0 - y2_out
con_y1 = y1_out - y1_in
con_y2 = y2_out - y2_in

Ins = [xx]

f_obj = aesara.function(Ins, obj)
f_con1 = aesara.function(Ins, con1)
f_con2 = aesara.function(Ins, con2,)
f_con_y1 = aesara.function(Ins, con_y1,)
f_con_y2 = aesara.function(Ins, con_y2,)

d_obj = aesara.function(Ins,    at.grad(obj, xx))
d_con1 = aesara.function(Ins,   at.grad(con1, xx))
d_con2 = aesara.function(Ins,   at.grad(con2, xx))
d_con_y1 = aesara.function(Ins, at.grad(con_y1, xx))
d_con_y2 = aesara.function(Ins, at.grad(con_y2, xx))

f_y1 = aesara.function(Ins, y1_out,)
f_y2 = aesara.function(Ins, y2_out,)
# problem:
# minimize obj wrt x1, z1, z2
# subject to con1 <= 0, con2 <= 0, 0 < x < 10, 0 < z1 < 10, 0 < z2 < 10

res = optimize.minimize(
    f_obj,
    [1.0, 5.0, 2.0, 1.0, 1.0],
    args=(),
    method="SLSQP",
    jac=d_obj,
    bounds=[(0, 10), (0, 10), (0, 10), (0, 10), (0, 10)],
    constraints=[
        {"type": "ineq", "fun": f_con1, "jac": d_con1},
        {"type": "ineq", "fun": f_con2, "jac": d_con2},
        {"type": "eq", "fun": f_con_y1, "jac": d_con_y1},
        {"type": "eq", "fun": f_con_y2, "jac": d_con_y2},
    ],
)
print(res)

print("checking...")
print("y1 =", f_y1(res.x),)
print("3.16 <= y1?", 3.16 <= res.x[3] )
print("y2 =", f_y2(res.x),)
print("y2 <= 24.0?", res.x[4] <= 24.0 )
