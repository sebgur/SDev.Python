import numpy as np
import pytest
from sdevpy.utilities.algos import unique_sorted, upper_bound


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


########## upper_bound ############################################################################

def test_upper_bound_exact_match():
    # Value exactly on a grid point should return that index
    arr = np.array([1.0, 2.0, 3.0])
    assert upper_bound(arr, 1.0) == 0
    assert upper_bound(arr, 2.0) == 1
    assert upper_bound(arr, 3.0) == 2


def test_upper_bound_strictly_between():
    # Value strictly between two grid points → index of the upper point
    arr = np.array([1.0, 2.0, 3.0])
    assert upper_bound(arr, 1.5) == 1
    assert upper_bound(arr, 2.5) == 2


def test_upper_bound_below_first():
    arr = np.array([1.0, 2.0, 3.0])
    assert upper_bound(arr, 0.5) == 0


def test_upper_bound_above_last():
    arr = np.array([1.0, 2.0, 3.0])
    assert upper_bound(arr, 4.0) == 3  # past-the-end index


def test_upper_bound_within_atol_of_previous():
    # Value within atol of arr[0] should snap down to index 0
    arr = np.array([1.0, 2.0, 3.0])
    # 1.0 + 1e-12 is within default atol=1e-11 of arr[0]=1.0
    # searchsorted returns 1, but snap should pull it back to 0
    assert upper_bound(arr, 1.0 + 1e-12) == 0


def test_upper_bound_outside_atol_of_previous():
    # Value outside atol should NOT snap; it belongs to the next interval
    arr = np.array([1.0, 2.0, 3.0])
    # 1.0 + 1e-9 is outside default atol=1e-11, so no snap
    assert upper_bound(arr, 1.0 + 1e-9) == 1


def test_upper_bound_custom_atol():
    arr = np.array([1.0, 2.0, 3.0])
    # With atol=1e-3, a value 5e-4 above arr[0] should snap to index 0
    assert upper_bound(arr, 1.0 + 5e-4, atol=1e-3) == 0
    # A value 2e-3 above arr[0] should NOT snap
    assert upper_bound(arr, 1.0 + 2e-3, atol=1e-3) == 1


def test_upper_bound_financial_times():
    # Realistic model time grid: daily to 2 years
    arr = np.array([1/365, 7/365, 30/365, 0.5, 1.0, 2.0])
    assert upper_bound(arr, 0.5) == 3
    assert upper_bound(arr, 1.0) == 4
    # Slightly above 0.5, outside tolerance → next slot
    assert upper_bound(arr, 0.5 + 1e-9) == 4


if __name__ == "__main__":
    print("Hello")
