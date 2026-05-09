import numpy as np
import numpy.typing as npt


def unique_sorted(arr: npt.ArrayLike, atol: float=1e-11) -> npt.ArrayLike:
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


def upper_bound(arr_sorted: npt.ArrayLike, value: float, atol: float=1e-11, clamp: bool=False) -> int:
    """ Upper bound index (with absolute tolerance):
        - when exactly on pillar: return that pillar's index
        - when between two pillars: return the upper pillar's index
        - before the first pillar: no ambiguity, return 0 (first pillar's index)
        - after the last pillar: return len(arr_sorted) for clamp=False, len(arr_sorted)-1 otherwise """
    idx = np.searchsorted(arr_sorted, value)

    # Check closeness with the previous pillar. If within tolerance, answer that pillar instead.
    if idx > 0: # No issue on first pillar as we can't be closer to the previous pillar
        if np.isclose(arr_sorted[idx - 1], value, rtol=0.0, atol=atol):
            idx = idx - 1

    if clamp:
        idx = min(idx, len(arr_sorted) - 1)

    return idx


def lower_bound(arr_sorted: npt.ArrayLike, value: float, atol: float = 1e-11, clamp: bool=False) -> int:
    """ Lower bound index (with absolute tolerance):
        - when exactly on pillar: return that pillar's index
        - when between two pillars: return the lower pillar's index
        - before the first pillar: return -1 for clamp=False, 0 for clamp=True (first pillar's index)
        - after the last pillar: no ambiguity, return len(arr_sorted)-1 """
    idx = np.searchsorted(arr_sorted, value, side='right') - 1

    # Check closeness with the next pillar. If within tolerance, answer that pillar instead.
    if idx + 1 < len(arr_sorted): # No issue on last pillar as we can't be closer to the next pillar
        if np.isclose(arr_sorted[idx + 1], value, rtol=0.0, atol=atol):
            idx = idx + 1

    if clamp:
        idx = max(idx, 0)

    return idx


if __name__ == "__main__":
    # Unique sorting
    arr = np.array([1.5, 2.3, 2.30001, 4.7, 2.29999, 5.1, 1.50001])
    unique = unique_sorted(arr, atol=0.001)
    print(unique)

    # Example sorted array
    print("<><><><><><><><>")
    arr = np.array([0.0, 1.0, 2.0])
    print(f"Array: {arr}")
    atol = 1e-4
    clamp = True

    values = [-0.5, 0.0, 0.001, 0.999, 1.0, 1.000001, 1.001, 1.5, 2.0, 2.000001, 2.5]

    # Locate
    for value in values:
        print("<>" * 20)
        # print(f"Point/Index(searchsorted): {value}/{np.searchsorted(arr, value)}")
        print(f"Point/Index(lower): {value}/{lower_bound(arr, value, atol, clamp=clamp)}")
        print(f"Point/Index(upper): {value}/{upper_bound(arr, value, atol, clamp=clamp)}")
