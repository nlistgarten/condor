import condor as co

class ShouldFail(co.ExplicitSystem):
    x = input()
    output.y = x**2
    z = placeholder()

sf = ShouldFail(3.0)

class Check(co.OptimizationProblem):
    print(objective.shape)
