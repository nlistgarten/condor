import condor as co

class OutputRefCheck(co.ExplicitSystem):
    x = input()
    output.y = x**2
    output.z = y+1

chk = OutputRefCheck(3.)


class ShouldFail(co.ExplicitSystem):
    x = input()
    output.y = x**2
    z = placeholder()

sf = ShouldFail(3.0)

class Check(co.OptimizationProblem):
    print(objective.shape)
