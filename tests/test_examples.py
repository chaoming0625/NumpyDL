# -*- coding: utf-8 -*-


from glob import glob
from importlib import import_module
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext
import sys

import pytest


EXAMPLES_DIR = join(dirname(dirname(__file__)), 'examples')


def _example_modules():
    paths = glob(join(EXAMPLES_DIR, "*py"))
    return [splitext(basename(path))[0] for path in paths]


@pytest.fixture
def example(request):
    sys.path.insert(0, EXAMPLES_DIR)
    request.addfinalizer(lambda: sys.path.remove(EXAMPLES_DIR))


@pytest.mark.slow
@pytest.mark.parametrize("module_name", _example_modules())
def test_example(example, module_name):
    for idx in range(10):
        try:
            main = getattr(import_module(module_name), "main{}".format('' if idx == 0 else idx))
            main(max_iter=1)  # run the example for one iteration
        except Exception as e:
            skip_exceptions = ['No module named',
                               'cannot import name',
                               'has no attribute']
            if any([text in str(e) for text in skip_exceptions]):
                continue
            else:
                raise

