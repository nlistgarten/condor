============================
Parallel Processing
============================

You should be able to treat models like any other function in terms of parallelization.
This example shows using the built-in :mod:`multiprocessing` to do thread-based
parallelization of an explicit system.

.. code-block:: python

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


    # produces...
    """
    ModelInput(x=1) ModelOutput(y=array([2.]))
    ModelInput(x=2) ModelOutput(y=array([1.]))
    ModelInput(x=3) ModelOutput(y=array([-2.]))
    """
