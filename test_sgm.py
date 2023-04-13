import numpy as np
from condor import Symbol, FreeComputation, MatchedComputation, Model, backend


#class DynamicsModel(metaclass=CondorModelType):
class DynamicsModel(Model):
    """
    DynamicsModel

    need input? (e.g., time varying paremeter) -> a parent class pushes input in
    or skip, assume plant always "pulls" input in

    in this context is "output" always point-wise in time? I think so, and it's a
    trajectory analysis model's outputs that define overall output

    fundamentally need to decide how different computations play. Maybe "output" is the
    base on CondorModel, and it is always the one that is returned by calling -- would
    be consistent way to pull controllers, measurements, etc from the plant model
    need API to specify which of the symbols are used as arguments to call?hkk

    then trajectory analysis model's IMP can 

    """
    print("starting app code for DynamicsModel class")

    # TODO: this needs its own descriptor type? OR do we want user classes to do
    # t = DynamicsModel.independent_variable ? That would allow 
    independent_variable = backend.symbol_generator('t')
    state = Symbol()
    parameter = Symbol()
    dot = MatchedComputation(state)
    output = FreeComputation()
    print("ending app code for DynamicsModel class")


class MySystem(DynamicsModel):
    print("starting user code for LTI class")
    n = 2
    m = 1
    x = state(n)
    C = state(n=n,m=n, symmetric=True)
    A = np.array([
        [0, 1],
        [0, 0],
    ])

    # A = parameter(n,n)
    # B = parameter(n,m)
    # K = parameter(m,n)


    # indexing an output/computation by state
    dot[x] = A@x #(A - B @ K) @ x
    dot[C] = A@C + C@A.T
    # indexing by integer/slice/could probably do advanced if needed
    dot[0] = A@A.T
    dot[:5] = A.T@A
    # dot naming, although have to do more work to minimize other setattr's 
    output.y = C.T @ C
    print("ending user code for LTI class")

