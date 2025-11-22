""" Utilities for solving tridiagonal systems """
import numpy as np
from scipy.linalg import solve_banded


def solve(upper, main, lower, y):
    n = main.shape[0]
    ab = np.zeros((3, n))
    ab[0, 1:] = upper
    ab[1, :] = main
    ab[2, :-1] = lower

    # Solve the system
    x = solve_banded((1, 1), ab, y)
    return x


if __name__ == "__main__":
    # Bands
    upper = np.array([-1, -1, -1.5, -1]) # n-1
    main = np.array([2, 2, 2, 2, 2]) # n
    lower = np.array([-1, -1, -1, -1]) # n-1

    # Input vector
    y = np.array([1, 2, 3, 4, 18])
    print(f"Input vector: {y}")

    # Run scipy.linalg.solve_banded
    x = solve(upper, main, lower, y)
    # n = main.shape[0]
    # ab = np.zeros((3, n))
    # ab[0, 1:] = upper
    # ab[1, :] = main
    # ab[2, :-1] = lower

    # # Solve the system
    # x = solve_banded((1, 1), ab, y)
    print(f"Solution: {x}")

    # Verify the solution
    A_full = np.diag(main) + np.diag(upper, 1) + np.diag(lower, -1)
    # print(A_full)
    print("Matrix")
    print(A_full)
    print("Verify by multiplication:")
    print(A_full @ x)
