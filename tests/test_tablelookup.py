import numpy as np

import condor as co


def test_table_optimization():
    data_yy = dict(
        sigma=np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.1833, 0.1621, 0.1429, 0.1256, 0.1101, 0.0966],
                [0.4, 0.3600, 0.3186, 0.2801, 0.2454, 0.2147, 0.1879],
                [0.6, 0.5319, 0.4654, 0.4053, 0.3526, 0.3070, 0.2681],
                [0.8, 0.6896, 0.5900, 0.5063, 0.4368, 0.3791, 0.3309],
                [1.0, 0.7857, 0.6575, 0.5613, 0.4850, 0.4228, 0.3712],
            ]
        ),
        sigstr=np.array(
            [
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.04, 0.7971, 0.9314, 0.9722, 0.9874, 0.9939, 0.9969],
                [0.16, 0.7040, 0.8681, 0.9373, 0.9688, 0.9839, 0.9914],
                [0.36, 0.7476, 0.8767, 0.9363, 0.9659, 0.9812, 0.9893],
                [0.64, 0.8709, 0.9338, 0.9625, 0.9778, 0.9865, 0.9917],
                [1.0, 0.9852, 0.9852, 0.9880, 0.9910, 0.9935, 0.9954],
            ]
        ),
    )
    data_xx = dict(
        xbbar=np.linspace(0, 1.0, data_yy["sigma"].shape[0]),
        xhbar=np.linspace(0, 0.3, data_yy["sigma"].shape[1]),
    )
    Table = co.TableLookup(data_xx, data_yy, 1)  # noqa: N806
    tt = Table(0.5, 0.5)
    print(tt.input, tt.output)

    class MyOpt(co.OptimizationProblem):
        xx = variable(warm_start=False)  # , lower_bound=0., upper_bound=1.0,)
        yy = variable(warm_start=False)  # , lower_bound=0., upper_bound=0.3,)
        # constraint(xx, lower_bound=0.)#, upper_bound=1.0,)
        # constraint(yy, lower_bound=0.)#, upper_bound=0.3,)
        interp = Table(xx, yy)
        objective = (interp.sigma - 0.2) ** 2 + (interp.sigstr - 0.7) ** 2

        class Options:
            exact_hessian = False
            exact_hessian = True
            print_level = 0

    opt0 = MyOpt()
    print("first call")
    hessian_iters0 = opt0.implementation.callback._stats["iter_count"]

    MyOpt.Options.exact_hessian = False

    opt1 = MyOpt()
    print("call w/o hessian")
    no_hessian_iters = opt1.implementation.callback._stats["iter_count"]

    MyOpt.Options.exact_hessian = True

    opt2 = MyOpt()
    print("with hessian again")
    hessian_iters1 = opt2.implementation.callback._stats["iter_count"]

    assert opt0.objective < 1e-19
    assert opt1.objective < 1e-19
    assert np.isclose(opt0.objective, opt1.objective)

    assert opt0.implementation.callback._stats["success"]
    assert opt1.implementation.callback._stats["success"]
    assert opt2.implementation.callback._stats["success"]

    assert no_hessian_iters > 7 * hessian_iters0
    assert hessian_iters0 == hessian_iters1
