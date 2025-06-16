"""
Warm Starting in Iterative Solvers
==================================
"""

import condor

# %%
# Here's a plain Python function implementing the well-known `Rosenbrock optimization
# benchmark function <https://en.wikipedia.org/wiki/Rosenbrock_function>`_.


def rosenbrock(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x**2) ** 2


# TODO plot the function and maybe iteration history?

# %%
# We'll minimize this function on a circle of a given radius and track the iteration
# history with an ``iter_callback`` option:

call_from_count = []


class RosenbrockOnCircle(condor.OptimizationProblem):
    r = parameter()
    x = variable(warm_start=False)
    y = variable(warm_start=False)

    objective = rosenbrock(x, y)

    constraint(x**2 + y**2 == r**2)

    class Options:
        print_level = 0

        @staticmethod
        def iter_callback(i, variable, objective, constraint):
            print(f"  inner {i:2d}: {variable=}")


# %%
# Calling the model twice without warm starting gives the same number of iterations each
# time.

out1 = RosenbrockOnCircle(r=2)

# %%
# Enabling warm start won't change the number of iterations on this call, but it will on
# the next call.

RosenbrockOnCircle.x.warm_start = True
RosenbrockOnCircle.y.warm_start = True

out2 = RosenbrockOnCircle(r=2)

# %%

out3 = RosenbrockOnCircle(r=2)

# %%

for use_warm_start in [False, True]:
    print("=== with warm_start =", use_warm_start)
    RosenbrockOnCircle.x.warm_start = use_warm_start
    RosenbrockOnCircle.y.warm_start = use_warm_start

    print("=== Embed within optimization over disk radius")

    class Outer(condor.OptimizationProblem):
        # r = variable(initializer=2+(5/16)+(1/64))
        r = variable(initializer=1.5, warm_start=False)

        out = RosenbrockOnCircle(r=r)

        objective = rosenbrock(out.x, out.y)

        class Options:
            print_level = 0

            # with exact_hessian = False means more outer iters and also a larger
            # percentage of calls correctly going through #the warm start -- I assume
            # the ones where it is re-starting is because of the jacobian?,
            # produces about a 16 iter difference
            exact_hessian = False

            @staticmethod
            def iter_callback(i, variable, objective, constraint):
                print(f"outer {i:2d}: {variable=}")

    out = Outer()
    print(out.r)
