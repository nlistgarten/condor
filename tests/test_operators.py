import pytest

import condor as co

backend = co.backend
ops = backend.operators


def test_min_max():
    class TestMax(co.ExplicitSystem):
        x = input()
        output.y = ops.max([0.3, x])
        output.z = ops.min([0.3, x])
        output.yy = ops.min(ops.concat([0.3, x]))
        output.zz = ops.max(ops.concat([0.3, x]))

    chk1 = TestMax(0.5)
    chk2 = TestMax(0.1)

    assert chk1.y == 0.5
    assert chk1.z == 0.3

    assert chk2.y == 0.3
    assert chk2.z == 0.1


def test_if_():
    class Check(co.ExplicitSystem):
        catd = input()
        output.emlf = ops.if_else(
            (catd == 0, 3.8),  # normal design FAR Part 23
            (catd == 1, 4.4),  # utility design FAR 23
            (catd == 2, 6.0),  # aerobatic design FAR 23
            (catd == 3, 2.5),  # transports FAR 25
            (catd > 3, catd),  # input design limit load factor
            1.234,  # else
        )

    assert Check(2.2).emlf == 1.234
    assert Check(1).emlf == 4.4
    assert Check(2).emlf == 6.0
    assert Check(12).emlf == 12

    with pytest.raises(ValueError, match="if_else requires an else_action"):

        class Check(co.ExplicitSystem):
            catd = input()
            output.emlf = ops.if_else(
                (catd == 0, 3.8),  # normal design FAR Part 23
                (catd == 1, 4.4),  # utility design FAR 23
                (catd == 2, 6.0),  # aerobatic design FAR 23
                (catd == 3, 2.5),  # transports FAR 25
                (catd > 3, catd),  # input design limit load factor
            )


def test_jacobian_empty():
    class TestJacobian(co.ExplicitSystem):
        x = input()

    ops.jacobian(TestJacobian.output.flatten(), TestJacobian.input.flatten())
