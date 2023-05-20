import numpy as np
import condor as co
import matplotlib.pyplot as plt
from sgm_test_util import LTI_plot
from scipy import linalg


class DblInt(co.ODESystem):
    A = np.array([
        [0, 1],
        [0, 0],
    ])
    B = np.array([[0,1]]).T
    x = state(shape=A.shape[0])
    t1 = parameter()
    t2 = parameter()
    # TODO: fix once mode and control work?
    dot[x] = A@x + B*(
        1*(t < t1)
        -1*(t >= t1)*(t<(t1+t2))
    )

class Switch1(DblInt.Event):
    function = t - t1

class Switch2(DblInt.Event):
    function = t - t2 - t1
    terminate = True

class Transfer(DblInt.TrajectoryAnalysis):
    initial[x] = [-9., 0.]
    Q = np.eye(2)
    cost = trajectory_output((x.T @ Q @ x)/2)
    tf = 10.


from condor.backends.casadi.implementations import OptimizationProblem
class MinimumTime(co.OptimizationProblem):
    t1 = variable(lower_bound=0)
    t2 = variable(lower_bound=0)
    objective = Transfer(t1, t2).cost
    """
    uncaled objective:
     fun: 9.936735272864849e-07
     jac: array([ 0.00248207, -0.00139785])
 message: 'Desired error not necessarily achieved due to precision loss.'
    nfev: 83
     nit: 9
    njev: 78
  status: 2
 success: False
       x: array([3.00007694, 2.99858558])


    scaled 100:
o=[[7.49673507e-09]]
computed jac: [[-3.45025525e-04 -9.77213736e-05]]

In [1]: opt._stats
Out[1]: 
     fun: 8.689026185891693e-07
     jac: array([ 0.04098872, -0.01228316])
 message: 'Desired error not necessarily achieved due to precision loss.'
    nfev: 70
     nit: 9
    njev: 66
  status: 2
 success: False
       x: array([3.00004197, 2.99985115])

ipopt wouldn't converge, really good targeting but jacobian never got small enough.
Maybe because using events instead of time special case?

args [DM([3.00004, 2.99998]), DM(00)]
p=[3.00004, 2.99998]
o=[[8.87272654e-10]]
computed jac: [[2.50302906e-04 2.12295932e-06]]
    """
    class Casadi(co.Options):
        exact_hessian = False
        method = OptimizationProblem.Method.scipy_cg

sim = Transfer(t1=1., t2= 4.,)
LTI_plot(sim)
#plt.show()
0.0169526, 3.76823
broke_ts = [0.01695263, 3.76822885]

#sim = Transfer(*broke_ts)
#jac_callback = sim.implementation.callback.jac_callback
#jac_callback(broke_ts, [0])
MinimumTime.implementation.set_initial(t1=2.163165480675697, t2=4.361971866705403)
opt = MinimumTime()

print(opt._stats)

