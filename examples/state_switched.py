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
    p1 = parameter()
    p2 = parameter()
    # TODO: fix once mode and control work?
    dot[x] = A@x + B*(
        1*(x[0] < p1)
        -1*(x[0] >= p1)*(x[0]<(p2))
    )

class Switch1(DblInt.Event):
    function = x[0] - p1

class Switch2(DblInt.Event):
    function = x[0] - p2
    terminate = True

class Transfer(DblInt.TrajectoryAnalysis):
    initial[x] = [-9., 0.]
    xd = [1., 2.]
    Q = np.eye(2)
    cost = trajectory_output(((x-xd).T @ Q @ (x-xd))/2)
    tf = 100.

    #class Casadi(co.Options):
        #use_lam_te_p = False
        #include_lam_dot = True

"""
xtep: 3.162277664286934 delta_fs [0.00010635, -1] delta_xs [7.16227766e+00 1.06350492e-04]
xtep: [-4.59824344  2.96405615] delta_fs [5.91527e-13, -2] delta_xs [5.89661653e-12 5.91526828e-13]
"""


from condor.backends.casadi.implementations import OptimizationProblem
class MinimumTime(co.OptimizationProblem):
    p1 = variable(lower_bound=0)
    p2 = variable(lower_bound=0)
    objective = Transfer(p1, p2).cost 

    class Casadi(co.Options):
        exact_hessian = False
        method = OptimizationProblem.Method.scipy_cg

p0 = -4., -1.
sim = Transfer(*p0)
LTI_plot(sim)
#plt.show()

MinimumTime.implementation.set_initial(p1=p0[0], p2=p0[1])
opt = MinimumTime()
print(opt._stats)

"""

lamda pdate - d2te, jac update +
In [1]: opt._stats
Out[1]: 
     fun: 2.4994146196380527
     jac: array([-2.61928816, -2.        ])
 message: 'Desired error not necessarily achieved due to precision loss.'
    nfev: 60
     nit: 0
    njev: 50
  status: 2
 success: False
       x: array([-4., -1.])

lamda pdate + d2te, jac update -
In [1]: opt._stats
Out[1]: 
     fun: 0.053750891069066484
     jac: array([-0.03535706, -1.26122735])
 message: 'Desired error not necessarily achieved due to precision loss.'
    nfev: 64
     nit: 1
    njev: 55
  status: 2
 success: False
       x: array([-4.62015512, -0.26122735])

both above with jac update using lamda te+ including terminal



NO d2te/d{}dt term at all works better?? T
with lamda after update:

In [1]: opt._stats
Out[1]: 
     fun: 0.6288633421574172
     jac: array([ 0.77463452, -0.29213221])
 message: 'Desired error not necessarily achieved due to precision loss.'
    nfev: 69
     nit: 2
    njev: 62
  status: 2
 success: False
       x: array([-3.94090216,  0.85393389])

Then with lamda before update:
In [1]: opt._stats
Out[1]: 
     fun: 0.6369777493361235
     jac: array([ 0.7540279, -0.3225361])
 message: 'Desired error not necessarily achieved due to precision loss.'
    nfev: 69
     nit: 2
    njev: 62
  status: 2
 success: False
       x: array([-3.93801297,  0.83873195])



didn't have right terminal cost, was trying to drive to origin:
fixing terminal coast, with lamda before update:
In [1]: opt._stats
Out[1]: 
     fun: 0.04568954566371569
     jac: array([-0.15051931,  0.00068665])
 message: 'Desired error not necessarily achieved due to precision loss.'
    nfev: 49
     nit: 1
    njev: 37
  status: 2
 success: False
       x: array([-3.32022966,  0.84949142])

lamda after update:
In [1]: opt._stats
Out[1]: 
     fun: 0.04568954568693802
     jac: array([-0.15051931,  0.00068665])
 message: 'Desired error not necessarily achieved due to precision loss.'
    nfev: 45
     nit: 1
    njev: 33
  status: 2
 success: False
       x: array([-3.32022966,  0.84949142])

"""

"""

lamda pdate - d2te, jac update +
In [1]: opt._stats
Out[1]: 
     fun: 2.4994146196380527
     jac: array([-2.61928816, -2.        ])
 message: 'Desired error not necessarily achieved due to precision loss.'
    nfev: 60
     nit: 0
    njev: 50
  status: 2
 success: False
       x: array([-4., -1.])

lamda pdate + d2te, jac update -
In [1]: opt._stats
Out[1]: 
     fun: 0.053750891069066484
     jac: array([-0.03535706, -1.26122735])
 message: 'Desired error not necessarily achieved due to precision loss.'
    nfev: 64
     nit: 1
    njev: 55
  status: 2
 success: False
       x: array([-4.62015512, -0.26122735])

both above with jac update using lamda te+ including terminal



NO d2te/d{}dt term at all works better?? T
with lamda after update:

In [1]: opt._stats
Out[1]: 
     fun: 0.6288633421574172
     jac: array([ 0.77463452, -0.29213221])
 message: 'Desired error not necessarily achieved due to precision loss.'
    nfev: 69
     nit: 2
    njev: 62
  status: 2
 success: False
       x: array([-3.94090216,  0.85393389])

Then with lamda before update:
In [1]: opt._stats
Out[1]: 
     fun: 0.6369777493361235
     jac: array([ 0.7540279, -0.3225361])
 message: 'Desired error not necessarily achieved due to precision loss.'
    nfev: 69
     nit: 2
    njev: 62
  status: 2
 success: False
       x: array([-3.93801297,  0.83873195])



didn't have right terminal cost, was trying to drive to origin:
fixing terminal coast, with lamda before update:
In [1]: opt._stats
Out[1]: 
     fun: 0.04568954566371569
     jac: array([-0.15051931,  0.00068665])
 message: 'Desired error not necessarily achieved due to precision loss.'
    nfev: 49
     nit: 1
    njev: 37
  status: 2
 success: False
       x: array([-3.32022966,  0.84949142])

lamda after update:
In [1]: opt._stats
Out[1]: 
     fun: 0.04568954568693802
     jac: array([-0.15051931,  0.00068665])
 message: 'Desired error not necessarily achieved due to precision loss.'
    nfev: 45
     nit: 1
    njev: 33
  status: 2
 success: False
       x: array([-3.32022966,  0.84949142])

"""

