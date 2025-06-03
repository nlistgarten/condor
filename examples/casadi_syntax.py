import casadi

from condor import CasadiFunctionCallback


def test_callbackfunc_2to1():
    x = casadi.SX.sym("x", 2)
    y = casadi.SX.sym("y")
    f = casadi.Function("f", [x, y], [x**2])
    cf = CasadiFunctionCallback(f)

    out = cf([0, 2], 1)

    assert out[0] == 0
    assert out[1] == 4


def test_callbackfunc_2to2():
    x = casadi.SX.sym("x", 2)
    y = casadi.SX.sym("y")
    f = casadi.Function("f", [x, y], [x**2, 2 * y])
    cf = CasadiFunctionCallback(f)

    out = cf([0, 2], 1)

    assert out[0][0] == 0
    assert out[0][1] == 4
    assert out[1] == 2


def test_callback_1to1():
    x = casadi.SX.sym("x")
    f = casadi.Function("f", [x], [x**2])
    cf = CasadiFunctionCallback(f)

    out = cf(2)

    assert out == 4


def test_callback():
    x = casadi.SX.sym("x", 3)
    f = casadi.Function("f", [x], [x**2])
    cf = CasadiFunctionCallback(f)
    print(cf.get_sparsity_in(0))

    out = cf([1, 2, 3])

    assert out[0] == 1
    assert out[1] == 4
    assert out[2] == 9
