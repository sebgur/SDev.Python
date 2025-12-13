import numpy as np


def unique_sorted(arr, atol=1e-11):
    """ Fast sorting and removal of duplicates """
    if len(arr) == 0:
        return arr

    # Sort
    sorted_arr = np.sort(arr)

    # Mark positions to keep: keep first element
    keep = np.ones(len(sorted_arr), dtype=bool)
    keep[0] = True

    # Keep elements where difference from previous > tolerance
    keep[1:] = np.abs(np.diff(sorted_arr)) > atol

    return sorted_arr[keep]


def upper_bound(arr_sorted, value, atol=1e-11):
    idx = np.searchsorted(arr_sorted, value)

    # The problem is that due to the float precision issue, we might answer the next
    # pillar even though we're within tolerance of the previous pillar.
    # So we check closeness with the previous pillar, and if we're within tolerance,
    # we answer that pillar instead.
    if idx > 0: # No issue on first pillar as we can't be closer to the previous pillar
        if np.isclose(arr_sorted[idx - 1], value, atol):
            idx = idx - 1

    return idx


if __name__ == "__main__":
    # Unique sorting
    arr = np.array([1.5, 2.3, 2.30001, 4.7, 2.29999, 5.1, 1.50001])
    unique = unique_sorted(arr, atol=0.001)
    print(unique)

    # Example sorted array
    print("<><><><><><><><>")
    arr = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    print(f"Array: {arr}")
    tol = 1e-4

    # Value to locate
    value = 0.5
    index = np.searchsorted(arr, value)
    print(f"Point/Index: {value}/{index}")
    index = upper_bound(arr, value, tol)
    print(f"Point/Index(new): {value}/{index}")

    value = 1.1
    index = np.searchsorted(arr, value)
    print(f"Point/Index: {value}/{index}")
    index = upper_bound(arr, value, tol)
    print(f"Point/Index(new): {value}/{index}")

    value = 1.5
    index = np.searchsorted(arr, value)
    print(f"Point/Index: {value}/{index}")
    index = upper_bound(arr, value, tol)
    print(f"Point/Index(new): {value}/{index}")

    value = 2.1999
    index = np.searchsorted(arr, value)
    print(f"Point/Index: {value}/{index}")
    index = upper_bound(arr, value, tol)
    print(f"Point/Index(new): {value}/{index}")

    value = 2.19999
    index = np.searchsorted(arr, value)
    print(f"Point/Index: {value}/{index}")
    index = upper_bound(arr, value, tol)
    print(f"Point/Index(new): {value}/{index}")

    value = 2.2
    index = np.searchsorted(arr, value)
    print(f"Point/Index: {value}/{index}")
    index = upper_bound(arr, value, tol)
    print(f"Point/Index(new): {value}/{index}")

    value = 2.2000001
    index = np.searchsorted(arr, value)
    print(f"Point/Index: {value}/{index}")
    index = upper_bound(arr, value, tol)
    print(f"Point/Index(new): {value}/{index}")

    value = 5.49999
    index = np.searchsorted(arr, value)
    print(f"Point/Index: {value}/{index}")
    index = upper_bound(arr, value, tol)
    print(f"Point/Index(new): {value}/{index}")

    value = 5.5
    index = np.searchsorted(arr, value)
    print(f"Point/Index: {value}/{index}")
    index = upper_bound(arr, value, tol)
    print(f"Point/Index(new): {value}/{index}")

    value = 5.5000001
    index = np.searchsorted(arr, value)
    print(f"Point/Index: {value}/{index}")
    index = upper_bound(arr, value, tol)
    print(f"Point/Index(new): {value}/{index}")

    value = 5.6
    index = np.searchsorted(arr, value)
    print(f"Point/Index: {value}/{index}")
    index = upper_bound(arr, value, tol)
    print(f"Point/Index(new): {value}/{index}")
