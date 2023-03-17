
class DynamicsModel(metaclass=CondorModelType):
    print("starting user code for CondorModel class")
    state = _condor_symbol_generator()
    parameter = _condor_symbol_generator()
    dot = _condor_computation(state)
    output = _condor_computation()
    print("ending user code for TrajectoryModel class")


class LTI(DynamicsModel):
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


