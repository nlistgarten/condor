import condor as co
import numpy as np
import pytest
backend = co.backend
ops = backend.operators


def test_output_ref():
    class TestMax(co.ExplicitSystem):
        x = input()
        output.y = ops.max([0.3, x])
        output.z = ops.min([0.3, x])

    chk1 = TestMax(0.5)
    chk2 = TestMax(0.1)

    assert chk1.y == 0.5
    assert chk1.z == 0.3

    assert chk2.y == 0.3
    assert chk2.z == 0.1
