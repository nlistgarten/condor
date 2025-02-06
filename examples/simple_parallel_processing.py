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
    with Pool(5) as p:
        models = p.map(Model, [1, 2, 3])

    for model in models:
        print(model.input, model.output)

