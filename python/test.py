import numpy as np

print("Hello World!")

t = [-2, -1, -3, 0, 2, 3]
print(t)
t = np.asarray(t)
t = np.reshape(t, (6, 1))
print(t.shape)
print(t)

t = np.maximum(t, 0.1)
print(t)

