# -*- coding: utf-8 -*-

import pytest
import numpy as np


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


def test_zero():
    from npdl.initialization import Zero

    sample = Zero().call((100, 20))
    assert sample.shape == (100, 20)
    assert sample.any() == 0.


def test_one():
    from npdl.initialization import One

    sample = One().call((10, 20))

    assert sample.shape == (10, 20)
    assert sample.any() == 1.


def test_normal():
    from npdl.initialization import Normal

    shape = (300, 400)
    sample = Normal(std=0.01, mean=0.0).call(shape)

    assert shape == sample.shape
    assert -0.001 < sample.mean() < 0.001
    assert 0.009 < sample.std() < 0.011


def test_uniform():
    from npdl.initialization import Uniform

    sample = Uniform(scale=0.1).call((200, 300))

    assert sample.shape == (200, 300)
    assert -0.1 < sample.min() < -0.09
    assert 0.09 < sample.max() < 0.1


def test_lecun_uniform():
    from npdl.initialization import LecunUniform

    shape = (200, 300)
    scale = np.sqrt(3 / 200)

    sample = LecunUniform().call(shape)

    assert sample.shape == shape
    assert - scale * 1.5 < sample.min() < -scale * 0.5
    assert 0.5 * scale < sample.max() < 1.5 * scale


def test_glorot_uniform():
    from npdl.initialization import GlorotUniform

    shape = (300, 400)
    scale = np.sqrt(6. / sum(shape))

    sample = GlorotUniform().call(shape)

    assert sample.shape == shape
    assert - 1.5 * scale < sample.min() < - 0.5 * scale
    assert 0.5 * scale < sample.max() < 1.5 * scale


def test_glorot_normal():
    from npdl.initialization import GlorotNormal

    shape = (300, 400)
    std = np.sqrt(2. / sum(shape))

    sample = GlorotNormal().call(shape)

    assert sample.shape == shape
    assert -0.001 < sample.mean() < 0.001
    assert std * 0.9 < sample.std() < std * 1.1


def test_he_uniform():
    from npdl.initialization import HeUniform

    shape = (300, 400)
    scale = np.sqrt(6. / shape[0])

    sample = HeUniform().call(shape)

    assert sample.shape == shape
    assert -1.5 * scale < sample.min() < - 0.5 * scale
    assert scale * 0.5 < sample.max() < scale * 1.5


def test_he_normal():
    from npdl.initialization import HeNormal

    shape = (300, 400)
    std = np.sqrt(6. / shape[0])

    sample = HeNormal().call(shape)

    assert sample.shape == shape
    assert -0.001 < sample.mean() < 0.001
    assert std * 0.5 < sample.std() < std * 1.5


def test_orthogonal_relu_gain():
    from npdl.initialization import Orthogonal

    assert Orthogonal('relu').gain == np.sqrt(2)


def test_orthogonal():
    from npdl.initialization import Orthogonal

    sample = Orthogonal().call((100, 200))
    assert np.allclose(np.dot(sample, sample.T), np.eye(100), atol=1e-6)

    sample = Orthogonal().call((200, 100))
    assert np.allclose(np.dot(sample.T, sample), np.eye(100), atol=1e-6)


def test_orthogonal_gain():
    from npdl.initialization import Orthogonal

    gain = 2
    sample = Orthogonal(gain).call((100, 200))
    assert np.allclose(np.dot(sample, sample.T), gain * gain * np.eye(100),
                       atol=1e-6)

    gain = np.sqrt(2)
    sample = Orthogonal('relu').call((100, 200))
    assert np.allclose(np.dot(sample, sample.T), gain * gain * np.eye(100),
                       atol=1e-6)


def test_orthoganal_multi():
    from npdl.initialization import Orthogonal

    sample = Orthogonal().call((100, 50, 80))
    sample = sample.reshape(100, 50 * 80)
    assert np.allclose(np.dot(sample, sample.T), np.eye(100), atol=1e-6)
