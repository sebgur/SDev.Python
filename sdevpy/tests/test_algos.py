import numpy as np
import pytest
from sdevpy.utilities.algos import unique_sorted, upper_bound, lower_bound


########## unique_sorted ##########################################################################

def test_unique_sorted_basic():
    arr = np.array([3.0, 1.0, 2.0, 1.0, 3.0])
    result = unique_sorted(arr)
    assert list(result) == [1.0, 2.0, 3.0]


def test_unique_sorted_already_unique():
    arr = np.array([1.0, 2.0, 3.0])
    result = unique_sorted(arr)
    assert list(result) == [1.0, 2.0, 3.0]


def test_unique_sorted_within_tolerance():
    # Two values within default atol=1e-11 should be collapsed to one
    arr = np.array([1.0, 1.0 + 1e-12, 2.0])
    result = unique_sorted(arr)
    assert len(result) == 2
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(2.0)


def test_unique_sorted_outside_tolerance():
    # Two values outside default atol=1e-11 should both be kept
    arr = np.array([1.0, 1.0 + 1e-10, 2.0])
    result = unique_sorted(arr)
    assert len(result) == 3


def test_unique_sorted_single_element():
    arr = np.array([42.0])
    result = unique_sorted(arr)
    assert list(result) == [42.0]


def test_unique_sorted_empty():
    arr = np.array([])
    result = unique_sorted(arr)
    assert len(result) == 0


def test_unique_sorted_all_identical():
    arr = np.array([5.0, 5.0, 5.0])
    result = unique_sorted(arr)
    assert list(result) == [5.0]


def test_unique_sorted_custom_atol():
    # With atol=0.01, values within 0.01 of each other collapse
    arr = np.array([1.0, 1.005, 2.0])
    result = unique_sorted(arr, atol=0.01)
    assert len(result) == 2

    # With atol=0.001, they are distinct
    result2 = unique_sorted(arr, atol=0.001)
    assert len(result2) == 3


########## upper/lower bounds #####################################################################
REF_VECTOR = np.array([0.0, 1.0, 2.0])
TEST_VECTOR = [-0.5, 0.0, 0.001, 0.999, 0.99999, 1.0, 1.000001, 1.001, 1.5, 2.0, 2.000001, 2.5]

def test_upper_bound_with_clamping():
    atol = 1e-4
    clamp = True
    test = []
    for value in TEST_VECTOR:
        test.append(upper_bound(REF_VECTOR, value, atol, clamp=clamp))

    ref = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    assert test == ref


def test_upper_bound_without_clamping():
    atol = 1e-4
    test = []
    for value in TEST_VECTOR:
        test.append(upper_bound(REF_VECTOR, value, atol))

    ref = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3]
    assert test == ref


def test_lower_bound_with_clamping():
    atol = 1e-4
    clamp = True
    test = []
    for value in TEST_VECTOR:
        test.append(lower_bound(REF_VECTOR, value, atol, clamp=clamp))

    ref = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    assert test == ref


def test_lower_bound_without_clamping():
    atol = 1e-4
    test = []
    for value in TEST_VECTOR:
        test.append(lower_bound(REF_VECTOR, value, atol))

    ref = [-1, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    assert test == ref


if __name__ == "__main__":
    print("Hello")
