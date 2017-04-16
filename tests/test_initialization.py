# -*- coding: utf-8 -*-

import pytest


def test_initializer_sample():
    from npdl.initialization import Initializer

    with pytest.raises(NotImplementedError):
        Initializer().call((100, 100))


def test_shape():
    from npdl.initialization import Initializer

    shape = (10, 20)
    # Assert that all `Initializer` sublasses return the shape that
    # we've asked for in `call`:
    for subclass in Initializer.__subclasses__():
        if len(subclass.__subclasses__()):
            for subsubclass in subclass.__subclasses__():
                assert subsubclass().call(shape).shape == shape
        else:
            assert subclass().call(shape).shape == shape


def test_specified_rng():
    from npdl.utils.random import get_rng
    from npdl.utils.random import set_rng
    from npdl.initialization import Normal
    from npdl.initialization import Uniform
    from npdl.initialization import GlorotNormal
    from npdl.initialization import GlorotUniform

    from numpy.random import RandomState
    from numpy import allclose

    shape = (10, 20)
    seed = 12345
    rng = get_rng()

    for test_cls in [Normal, Uniform, GlorotNormal, GlorotUniform]:
        set_rng(RandomState(seed))
        sample1 = test_cls().call(shape)
        set_rng(RandomState(seed))
        sample2 = test_cls().call(shape)
        # reset to original RNG for other tests
        set_rng(rng)
        assert allclose(sample1, sample2), \
            "random initialization was inconsistent " \
            "for {}".format(test_cls.__name__)

