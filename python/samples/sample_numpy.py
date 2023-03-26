import numpy as np
from scipy.stats import norm
from random import randint

np.random.seed(42)

# Correlation with Cholesky
correl = [[1, 0.4], [0.4, 1]]
chol = np.linalg.cholesky(correl)
chol_t = np.transpose(chol)
print('cholesky')
print(chol)
print(chol_t)
print(np.matmul(chol, chol_t))

# Normal density
print()
print("Normal density")
scale = 1.0
mean = 0.0
print(norm.cdf(2.5, loc=mean, scale=scale))
print(norm.ppf(0.95, loc=mean, scale=scale))

# Maximum of two vectors
print()
print("Maximum of two vectors")
a = [0, 1, 2]
print(a)
b = [1, -1, 3]
print(b)
print(np.maximum(a, b))


# Create a list of random integers
print()
# print('Random permutation')
rand_integers = []
for i in range(10):
    rand_integers.append(randint(0, 729145))
print(rand_integers)
# print(indices)

# Ceil function
alpha = [-0.6, -0.5, -0.2, 0.0, 0.1, 0.5, 0.99, 1.0, 1.1, 1.7]
beta = np.ceil(alpha)
# print(beta)

# Sort
unsorted = [[1, 2, 3], [5, 4, 3], [3, 2, 5]]
# print(np.sort(unsorted, axis=1))

# Multi-dimensional arrays
a = np.ndarray(shape=(2, 2))
b = np.zeros(shape=(2, 3))
print(b)
s = b.shape
counter = 0
for i in range(s[0]):
    for j in range(s[1]):
        b[i, j] = counter
        counter = counter + 1
print(b)

# Matrices
print()
print("Matrices")
m = np.ndarray(shape=(3, 3))
for i in range(3):
    for j in range(3):
        m[i, j] = i + 1
print(m)
col_sums = m.sum(axis=0, keepdims=True)
print(col_sums)
row_sums = m.sum(axis=1, keepdims=True)
print(row_sums)
print(m / row_sums)

# Numpy weird syntax with second set of brackets
x = np.ndarray(shape=(3, 2))
x[0] = [1, 2]
x[1] = [3, 4]
x[2] = [5, 6]
y = np.ndarray(shape=(3,))
y[0] = 1
y[1] = 2
y[2] = 2
print(x[:, 0][y == 2])
