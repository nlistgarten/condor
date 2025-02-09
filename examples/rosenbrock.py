import condor

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2) **2


class RosenbrockOnCircle(condor.OptimizationProblem):
    r = parameter()
    x = variable(warm_start=False)
    y = variable(warm_start=False)

    objective = rosenbrock(x, y)

    constraint(x**2 + y**2 == r**2)

    class Options:
        print_level = 0

        @staticmethod
        def init_callback(parameter, opts):
            print("  inner init:", parameter)

        @staticmethod
        def iter_callback(i, variable, objective, constraint):
            print("  inner: ", i, variable, objective)


print("=== Call twice, should see same iters")
out1 = RosenbrockOnCircle(r=2)#**0.5)
print("---")

RosenbrockOnCircle.x.warm_start = True
RosenbrockOnCircle.y.warm_start = True

out2 = RosenbrockOnCircle(r=2)#**0.5)

print(3*"\n")

print("=== From warm start")
out3 = RosenbrockOnCircle(r=2)#**0.5)

print(3*"\n")

#print("=== Set warm start, should see fewer iters on second call")
#RosenbrockOnCircle.x.warm_start = True
#RosenbrockOnCircle.y.warm_start = True
#RosenbrockOnCircle(r=2)
#print("---")
#RosenbrockOnCircle(r=2)
#
#print(3*"\n")



for use_warm_start in [False, True]:
    print("=== with warm_start =",use_warm_start)
    RosenbrockOnCircle.x.warm_start = use_warm_start
    RosenbrockOnCircle.y.warm_start = use_warm_start


    print("=== Embed within optimization over disk radius")

    class Outer(condor.OptimizationProblem):
        #r = variable(initializer=2+(5/16)+(1/64))
        r = variable(initializer=1.5, warm_start=False)

        out = RosenbrockOnCircle(r=r)

        objective = rosenbrock(out.x, out.y)

        class Options:
            print_level = 0
            exact_hessian = False
            # with exact_hessian = False means more outer iters and also a larger
            # percentage of calls correctly going through #the warm start -- I assume
            # the ones where it is re-starting is because of the jacobian?, 
            # produces about a 16 iter difference

            @staticmethod
            def init_callback(parameter, opts):
                print("outer init:", parameter)

            @staticmethod
            def iter_callback(i, variable, objective, constraint):
                print("outer: ", i, variable, objective)



    out = Outer()
    print(out.r)
    #break
