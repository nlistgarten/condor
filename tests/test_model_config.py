import numpy as np
import pytest

import condor as co


def test_model_config():
    a = np.array([[0, 1], [0, 0]])
    b = np.array([[0], [1]])

    ct_mod = co.settings.get_module("modules.configured_model", a=a, b=b)
    dbl_int = ct_mod.LTI
    # no events
    assert len(dbl_int.Event._meta.subclasses) == 0

    sp_mod = co.settings.get_module("modules.configured_model", a=a, b=b, dt=0.5)
    sp_dbl_int = sp_mod.LTI
    # one DT event
    assert len(sp_dbl_int.Event._meta.subclasses) == 1

    assert dbl_int is not sp_dbl_int

    with pytest.raises(ValueError, match="Extra keyword arguments"):
        co.settings.get_module("modules.configured_model", a=a, b=b, extra="something")
