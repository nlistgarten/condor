import condor

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2) **2

call_from_count = []
param_count = []

# these two callback data arrays end up being 7 elements long,
# 3 calls as a top level (1 with warm start off, 2 with warm start on)
# then Outer gets called twice (once with warm start off, once off)
# each outer results in two elements to the callback data arrays, 
# 1 is used by outer to solve the embedded problem (initiated with a variable, but then
# called at least once for each iter), which only goes through casadi infrastructure as
# Callback, and doesn't hit implementation after binding the variable
# 1 is used by the bind_embedded_models with the final solution, which is essentially
# "top level" and goes through implementation
# currently not able to link the final solution of the embedded to the implementation --
# might be able to fix that?
# definitely have access to the embedded solution, in bind_embedded_model, the
# embedded_model.implementation.callback definitely has the correct last_x. e.g., line
# 1333 of models.py
# need an API for using the callback effectively -- probably applies to SGM as well.
# maybe if the implementation and model API was more defined, with more hooks for the
# call process.

# ok, figured out a decent API I think. iter_count for the two bind_embedded_model calls
# is 0. can use call_from_count [-4], [-2] to show warm start worked?


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
            call_from_count.append(0)
            param_count.append(parameter)

        @staticmethod
        def iter_callback(i, variable, objective, constraint):
            print("  inner: ", i, variable, objective)
            call_from_count[-1] += 1


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
