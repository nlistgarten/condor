import numpy as np
import pytest

import condor as co


def test_model_config():
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])

    ct_mod = co.settings.get_module("modules.configured_model", A=A, B=B)
    dbl_int = ct_mod.LTI
    # no events
    assert len(dbl_int.Event._meta.subclasses) == 0

    sp_mod = co.settings.get_module("modules.configured_model", A=A, B=B, dt=0.5)
    sp_dbl_int = sp_mod.LTI
    # one DT event
    assert len(sp_dbl_int.Event._meta.subclasses) == 1

    assert dbl_int is not sp_dbl_int

    with pytest.raises(ValueError):
        co.settings.get_module("modules.configured_model", A=A, B=B, extra="something")
