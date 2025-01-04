"""
============================
Parallel Processing
============================
"""
# %%
# In Python, parallel processing requires a call-able that returns fully pickle-able
# results. The instance of a Condor model is not pickle-able so it causes an error when
# used directly.

import condor
class Model(condor.ExplicitSystem):
    x = input()
    output.y = -x**2 + 2*x + 1


from multiprocessing import Pool
if __name__ == "__main__":
    try:
        with Pool(5) as p:
            print(p.map(Model, [1, 2, 3]))
    except Exception as e:
        print("Error when trying to return a model instance in parallel")
        print(e)
    else:
        raise ValueError("expected an error")

# %%
# a simple wrapper that extracts the fields, which are simple dataclasses, can be used
# instead.
def wrapper(*args, **kwargs):
    m = Model(*args, **kwargs)
    return m.input, m.output

if __name__ == "__main__":
    with Pool(5) as p:
        print(p.map(wrapper, [1, 2, 3]))
