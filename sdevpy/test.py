""" Just to test things """
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import json


# Test location

# Example sorted array
arr = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
print(f"Array: {arr}")

# Sort arra with np.sort(arr)

# Value to locate
value = 0.5
index = np.searchsorted(arr, value)
print(f"Point/Index: {value}/{index}")

value = 1.1
index = np.searchsorted(arr, value)
print(f"Point/Index: {value}/{index}")

value = 1.5
index = np.searchsorted(arr, value)
print(f"Point/Index: {value}/{index}")

value = 2.1999
index = np.searchsorted(arr, value)
print(f"Point/Index: {value}/{index}")

value = 2.2
index = np.searchsorted(arr, value)
print(f"Point/Index: {value}/{index}")

value = 2.2000001
index = np.searchsorted(arr, value)
print(f"Point/Index: {value}/{index}")

value = 5.49999
index = np.searchsorted(arr, value)
print(f"Point/Index: {value}/{index}")

value = 5.5
index = np.searchsorted(arr, value)
print(f"Point/Index: {value}/{index}")

value = 5.5000001
index = np.searchsorted(arr, value)
print(f"Point/Index: {value}/{index}")

# print(f"Insert {value} at index {index} to keep the array sorted.")