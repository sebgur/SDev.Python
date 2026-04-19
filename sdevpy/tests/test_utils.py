import numpy as np
from sdevpy.utilities import tools


######## isequal ##################################################################################

def test_isequal_float_scalars():
    assert tools.isequal(1.0, 1.0 + 1e-13)

def test_isequal_float_scalars_within_custom_tol():
    assert tools.isequal(1.0, 1.05, tol=0.1)

def test_isequal_float_scalars_outside_tol():
    assert not tools.isequal(1.0, 1.05, tol=1e-6)

def test_isequal_float_scalars_exactly():
    assert tools.isequal(0.0, 0.0)

def test_isequal_float_lists():
    assert tools.isequal([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

def test_isequal_float_lists_within_tol():
    assert tools.isequal([1.0, 2.0], [1.0 + 1e-13, 2.0 - 1e-13])

def test_isequal_float_lists_outside_tol():
    assert not tools.isequal([1.0, 2.0], [1.0, 2.1])

def test_isequal_float_numpy_arrays():
    assert tools.isequal(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

def test_isequal_float_numpy_array_within_tol():
    assert tools.isequal(np.array([1.0, 2.0]), np.array([1.0 + 1e-13, 2.0]))

def test_isequal_int_scalars():
    assert tools.isequal(3, 3)

def test_isequal_int_scalars_not():
    assert not tools.isequal(3, 4)

def test_isequal_int_lists():
    assert tools.isequal([1, 2, 3], [1, 2, 3])

def test_isequal_int_lists_not():
    assert not tools.isequal([1, 2, 3], [1, 2, 4])

def test_isequal_int_numpy_arrays():
    assert tools.isequal(np.array([1, 2]), np.array([1, 2]))

def test_isequal_mixed_int_float():
    assert tools.isequal(1, 1.0)

def test_isequal_mixed_int_float_within_tol():
    assert tools.isequal(1, 1.0 + 1e-13)

def test_isequal_different_lengths_not():
    assert not tools.isequal([1.0, 2.0], [1.0, 2.0, 3.0])

def test_isequal_empty_lists():
    assert tools.isequal([], [])

######## Others ###################################################################################

def test_isiterable():
    test = [tools.isiterable(0.5), tools.isiterable("alpha"),
            tools.isiterable([1, 2]), tools.isiterable(np.asarray([0.1]))]
    ref = [False, True, True, True]
    assert test == ref
