import numpy as np
from sdevpy.tools import utils


def test_isiterable():
    test = [utils.isiterable(0.5), utils.isiterable("alpha"),
            utils.isiterable([1, 2]), utils.isiterable(np.asarray([0.1]))]
    ref = [False, True, True, True]
    assert test == ref
