# -*- coding: utf-8 -*-

import pytest


def test_objective():
    from npdl.objectives import Objective

    obj = Objective()

    with pytest.raises(NotImplementedError):
        obj.forward(None, None)

    with pytest.raises(NotImplementedError):
        obj.backward(None, None)

    assert str(obj) == 'Objective'




